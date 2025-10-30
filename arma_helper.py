# arma_feature_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ARMAFeatureBlock(nn.Module):

    def __init__(self, H=24, K_ar=5, K_ma=5, use_bias=True,
                 gate="softmax", init_zero_delta=True, norm_type="layer"):
        super().__init__()
        assert K_ar >= 1 and K_ar <= H
        assert K_ma >= 1 and K_ma <= H
        self.H = H
        self.K_ar = K_ar
        self.K_ma = K_ma
        self.use_bias = use_bias
        self.gate = gate
        self.init_zero_delta = init_zero_delta

        self.ar_kernel = nn.Parameter(torch.zeros(K_ar))
        self.ma_kernel = nn.Parameter(torch.zeros(K_ma))

        self.Gamma = nn.Linear(H, H, bias=False)

        if gate == "softmax":
            self.alpha_logits = nn.Parameter(torch.zeros(2))
        elif gate == "sigmoid":
            self.alpha_ar = nn.Parameter(torch.tensor(0.0))
            self.alpha_ma = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError("gate must be 'softmax' or 'sigmoid'")

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(H))
        else:
            self.register_parameter("bias", None)

        if norm_type == "layer":
            self.norm = nn.LayerNorm(H)
        elif norm_type == "batch":
            self.norm = None
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError("norm_type in {None, 'layer'} recommended")

        if init_zero_delta:
            nn.init.zeros_(self.ar_kernel)
            nn.init.zeros_(self.ma_kernel)
            nn.init.zeros_(self.Gamma.weight)
            if use_bias:
                nn.init.zeros_(self.bias)

    @staticmethod
    def _causal_conv_1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        B, H = x.shape
        x1 = x.unsqueeze(1)
        K = w.numel()
        x_pad = F.pad(x1, (K - 1, 0))
        y = F.conv1d(x_pad, w.view(1, 1, K))
        return y.squeeze(1)

    @staticmethod
    def _anti_causal_conv_1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        B, H = x.shape
        x1 = x.unsqueeze(1)
        K = w.numel()
        x_pad = F.pad(x1, (0, K - 1))
        y = F.conv1d(x_pad, w.view(1, 1, K))
        return y.squeeze(1)

    def forward(self, h_prev: torch.Tensor, h_cur: torch.Tensor):

        assert h_prev.shape == h_cur.shape and h_cur.shape[-1] == self.H
        B, H = h_cur.shape

        ar_term = self._causal_conv_1d(h_cur, self.ar_kernel)

        gamma_hprev = self.Gamma(h_prev)
        e_t = h_cur - gamma_hprev
        ma_term = self._anti_causal_conv_1d(e_t, self.ma_kernel)

        if self.gate == "softmax":
            a_ar, a_ma = torch.softmax(self.alpha_logits, dim=0)
        else:
            a_ar, a_ma = torch.sigmoid(self.alpha_ar), torch.sigmoid(self.alpha_ma)

        delta = a_ar * ar_term + a_ma * ma_term
        if self.use_bias:
            delta = delta + self.bias

        out = h_cur + (self.norm(delta) if self.norm is not None else delta)

        aux = {
            "alpha_ar": float(a_ar.detach()),
            "alpha_ma": float(a_ma.detach()),
            "ar_kernel": self.ar_kernel.detach().cpu(),
            "ma_kernel": self.ma_kernel.detach().cpu(),
        }
        return out, delta, aux


import torch
import torch.nn as nn
import torch.nn.functional as F


class ARMAFeatureBlockSafeMPS(nn.Module):

    def __init__(self, H=24, K_ar=5, K_ma=5, gate="softmax",
                 use_bias=True, init_zero_delta=True, norm_type="layer"):
        super().__init__()
        assert 1 <= K_ar <= H and 1 <= K_ma <= H
        self.H = H
        self.K_ar = K_ar
        self.K_ma = K_ma
        self.gate = gate
        self.use_bias = use_bias

        self.ar_kernel = nn.Parameter(torch.zeros(K_ar))
        self.ma_kernel = nn.Parameter(torch.zeros(K_ma))

        self.Gamma = nn.Linear(H, H, bias=False)

        if gate == "softmax":
            self.alpha_logits = nn.Parameter(torch.zeros(2))
        elif gate == "sigmoid":
            self.alpha_ar = nn.Parameter(torch.tensor(0.0))
            self.alpha_ma = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError("gate must be 'softmax' or 'sigmoid'")

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(H))
        else:
            self.register_parameter("bias", None)

        self.norm = nn.LayerNorm(H) if norm_type == "layer" else None

        ar_base = torch.zeros(H, H)
        for i in range(H):
            j0 = max(0, i - (K_ar - 1))
            ar_base[i, j0:i+1] = 1.0
        self.register_buffer("ar_base", ar_base)

        ma_base = torch.zeros(H, H)
        for i in range(H):
            j1 = min(H, i + K_ma)
            ma_base[i, i:j1] = 1.0
        self.register_buffer("ma_base", ma_base)

        if init_zero_delta:
            nn.init.zeros_(self.Gamma.weight)

    def _build_ar_mat(self, device=None, dtype=None):

        H, K = self.H, self.K_ar
        M = torch.zeros(H, H, device=device, dtype=dtype)
        for d in range(K):
            M += torch.diag(torch.ones(H - d, device=device, dtype=dtype), diagonal=-d) * self.ar_kernel[d]
        return M

    def _build_ma_mat(self, device=None, dtype=None):
        H, K = self.H, self.K_ma
        M = torch.zeros(H, H, device=device, dtype=dtype)
        for d in range(K):
            M += torch.diag(torch.ones(H - d, device=device, dtype=dtype), diagonal= d) * self.ma_kernel[d]
        return M

    def forward(self, h_prev: torch.Tensor, h_cur: torch.Tensor):
        assert h_prev.shape == h_cur.shape and h_cur.shape[-1] == self.H
        B, H = h_cur.shape
        device, dtype = h_cur.device, h_cur.dtype

        A_ar = self._build_ar_mat(device=device, dtype=dtype)
        A_ma = self._build_ma_mat(device=device, dtype=dtype)

        ar_term = h_cur @ A_ar.T

        gamma_hprev = self.Gamma(h_prev)
        e_t = h_cur - gamma_hprev

        ma_term = e_t @ A_ma.T

        if self.gate == "softmax":
            a_ar, a_ma = torch.softmax(self.alpha_logits, dim=0)
        else:
            a_ar, a_ma = torch.sigmoid(self.alpha_ar), torch.sigmoid(self.alpha_ma)

        delta = a_ar * ar_term + a_ma * ma_term
        if self.use_bias:
            delta = delta + self.bias

        out = h_cur + (self.norm(delta) if self.norm is not None else delta)

        aux = {
            "alpha_ar": float(a_ar.detach()),
            "alpha_ma": float(a_ma.detach()),
            "ar_kernel": self.ar_kernel.detach().cpu(),
            "ma_kernel": self.ma_kernel.detach().cpu(),
        }
        return out, delta, aux
