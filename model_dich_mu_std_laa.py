
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Optional, Literal, Tuple, List
from dataclasses import dataclass




def build_model_dp(model_cls, *args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(*args, **kwargs).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[DP] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model

def save_state(model, path):
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(sd, path)

def load_state(model, path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(sd)


def append_fourier_features(df_wide, K, period_days,
                            mode="vector",
                            start_idx=None):
    if K <= 0:
        return df_wide, 0

    existing = [c for c in df_wide.columns if c.startswith("x_") and c != "x_0"]
    if start_idx is None:
        start_idx = 1 + len(existing)

    t0 = pd.to_datetime(df_wide["day"]).astype("int64") // 86_400_000_000_000
    t_day = (t0 - t0.min()).astype(float).to_numpy()
    P = float(period_days)

    N = len(df_wide);
    H = 24

    if mode == "vector":
        feats = np.empty((N, 2 * K), dtype=np.float64)
        for k in range(1, K + 1):
            w = 2 * np.pi * k / P
            feats[:, 2 * (k - 1)] = np.sin(w * t_day)
            feats[:, 2 * (k - 1) + 1] = np.cos(w * t_day)
        cols = [f"x_{start_idx + j}" for j in range(feats.shape[1])]
        add = pd.DataFrame(feats, columns=cols, index=df_wide.index)
        return pd.concat([df_wide, add], axis=1), feats.shape[1]

    elif mode == "matrix":
        t_mat = t_day[:, None] + np.arange(H)[None, :] / 24.0
        S = np.stack([np.sin(2 * np.pi * k / P * t_mat) for k in range(1, K + 1)], axis=0)
        C = np.stack([np.cos(2 * np.pi * k / P * t_mat) for k in range(1, K + 1)], axis=0)

        blocks = []
        for h in range(H):
            blk = np.empty((N, 2 * K), dtype=np.float64)
            blk[:, 0::2] = S[:, :, h].T
            blk[:, 1::2] = C[:, :, h].T
            blocks.append(blk)
        Xf = np.concatenate(blocks, axis=1)
        cols = [f"x_{start_idx + j}" for j in range(Xf.shape[1])]
        add = pd.DataFrame(Xf, columns=cols, index=df_wide.index)
        return pd.concat([df_wide, add], axis=1), Xf.shape[1]

    else:
        raise ValueError("mode must be 'vector' or 'matrix'")

def simple_dst_fix(df: pd.DataFrame, start_at_midnight: bool = True) -> pd.DataFrame:

    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    df = df[~df["ds"].duplicated(keep="first")]

    start = df["ds"].iloc[0]
    if start_at_midnight:
        start = start.normalize()
    end = df["ds"].iloc[-1]
    full_idx = pd.date_range(start, end, freq="h")

    out = df.set_index("ds").reindex(full_idx)

    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].ffill()

    if out[num_cols].isna().any().any():
       out[num_cols] = out[num_cols].bfill()

    out = out.rename_axis("ds").reset_index()

    return out



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict




class DirichletMeanConcentration(nn.Module):

    def __init__(
        self,
        K: int = 24,
        d_model: int = 128,
        d_x: int = 0,
        nhead: int = 4,
        dropout: float = 0.0,
        activation: str = "tanh",
        tie_c_to_h_only: bool = True,   # True: c 仅由 h_t 决定；False: 允许 x_t 进入 c
        eps_alpha: float = 1e-6,
        use_gate=True,
    ):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.d_x = d_x
        self.eps_alpha = eps_alpha
        self.tie_c_to_h_only = tie_c_to_h_only


        # --- State update: GRU + SelfAttention + Linear ---
        self.rnn  = nn.GRU(input_size=K, hidden_size=d_model, batch_first=True)

        self.feat_embed = nn.Parameter(torch.randn(self.K, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        self.use_gate = use_gate


        self.to_h = nn.Linear(d_model, d_model)
        if activation == "tanh":
            self.h_act = nn.Tanh()
        elif activation == "relu":
            self.h_act = nn.ReLU(inplace=True)
        else:
            raise ValueError("activation must be 'tanh' or 'relu'.")

        # --- Decoder for π_t (mean direction) ---
        self.W_h = nn.Linear(d_model, K, bias=True)                 # hidden -> logits
        self.W_x = nn.Linear(d_x,     K, bias=False) if d_x > 0 else None

        # --- Concentration head for c_t (confidence) ---
        self.wc_h = nn.Linear(d_model, 1, bias=True)
        self.wc_x = nn.Linear(d_x,     1, bias=False) if (d_x > 0 and not tie_c_to_h_only) else None

    def _feature_attend(self, z):
        tokens = z.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.ffn(tokens))
        delta = self.out_proj(tokens).squeeze(-1)
        return z + (self.gate(z) * delta if self.use_gate else delta)

    # ---------- 1) State update ----------
    def encode_history(self, y_hist_std: torch.Tensor) -> torch.Tensor:
        """
        y_hist_std: [B, T_hist, K]
        return: h_t ∈ ℝ^{B×d_model}
        """
        h_seq, _ = self.rnn(y_hist_std)     # [B, T_hist, d_model]
        ctx = self.attn(h_seq)              # [B, T_hist, d_model]
        h_t = self.to_h(ctx[:, -1, :])      # 取“当前步”的上下文并线性映射
        return self.h_act(h_t)

    def _dirichlet_params(
        self,
        h_t: torch.Tensor,                   # [B, d_model]
        x_t: Optional[torch.Tensor] = None,  # [B, d_x]
    ):
        # logits -> π
        eta = self.W_h(h_t)                  # [B, K]
        if self.W_x is not None and x_t is not None:
            eta = eta + self.W_x(x_t)        # [B, K]
        pi = torch.softmax(eta, dim=-1)      # [B, K]

        # concentration c
        c_in = self.wc_h(h_t)                # [B, 1]
        if self.wc_x is not None and x_t is not None:
            c_in = c_in + self.wc_x(x_t)     # [B, 1]
        c = F.softplus(c_in) + 1e-6          # strictly positive

        # alpha = c * pi
        alpha = (c * pi).clamp_min(self.eps_alpha)  # [B, K]
        return alpha, pi, c

    # ---------- 3) Forward ----------
    @torch.no_grad()
    def _dirichlet_mean(self, alpha: torch.Tensor) -> torch.Tensor:
        """E[s] for Dir(alpha) = alpha / alpha.sum(-1, keepdim=True)"""
        return alpha / alpha.sum(dim=-1, keepdim=True)

    import torch

    def laa_dirichlet(self,
                      s: torch.Tensor,  # [B,K] 份额（可为均值或样本，不要求和=1，但建议接近）
                      alpha: torch.Tensor,  # [B,K] Dirichlet 参数
                      total: torch.Tensor,  # [B] 或 [B,1]
                      eps: float = 1e-12,
                      clamp_nonneg: bool = False):
        """
        Linear Accuracy Adjustment for Dirichlet-based share predictions.

        返回:
          y_adj: [B,K] 经过 LAA 的预测量 (单位与 total 一致)
          extras: 诊断信息(dict)：lambda, residual, Sigma_y
        """
        assert s.dim() == 2 and alpha.dim() == 2 and s.shape == alpha.shape
        B, K = s.shape
        if total.dim() == 1:
            total = total.unsqueeze(-1)  # [B,1]
        else:
            total = total[:, :1]  # [B,1]

        # 基础预测（未校准）
        y_hat = total * s  # [B,K]

        # Dirichlet 协方差（份额层面）：Σ_s = (diag(α) - αα^T / α0) / (α0(α0+1))
        alpha0 = alpha.sum(-1, keepdim=True)  # [B,1]
        a0 = alpha0.unsqueeze(-1)  # [B,1,1]
        diag = torch.diag_embed(alpha)  # [B,K,K]
        outer = alpha.unsqueeze(-1) * alpha.unsqueeze(-2)  # [B,K,K]
        Sigma_s = (diag - outer / a0) / (a0 * (a0 + 1.0))  # [B,K,K]

        # 量纲变换：Σ_y = total^2 * Σ_s
        t2 = (total ** 2).unsqueeze(-1)  # [B,1,1]
        Sigma_y = Sigma_s * t2  # [B,K,K]

        # 计算 λ = Σ_y 1 / (1' Σ_y 1)
        one = torch.ones(B, K, 1, device=s.device, dtype=s.dtype)  # [B,K,1]
        Sigma1 = torch.bmm(Sigma_y, one)  # [B,K,1]
        denom = torch.bmm(one.transpose(1, 2), Sigma1).squeeze(-1).squeeze(-1)  # [B]
        denom = denom + eps
        lam = (Sigma1.squeeze(-1) / denom.unsqueeze(-1))  # [B,K]

        # 残差与线性校准
        residual = total - y_hat.sum(-1, keepdim=True)  # [B,1]
        y_adj = y_hat + residual * lam  # [B,K]

        # （可选）保证非负并严格保总量
        if clamp_nonneg:
            y_adj = torch.clamp(y_adj, min=0)
            s_tmp = y_adj / y_adj.sum(-1, keepdim=True).clamp_min(eps)
            y_adj = total * s_tmp

        return y_adj, {"lambda": lam, "residual": residual, "Sigma_y": Sigma_y}

    def forward(
        self,
        y_hist_std: torch.Tensor,                 # [B, T_hist, K]
        x_t: Optional[torch.Tensor] = None,       # [B, d_x]
        total: Optional[torch.Tensor] = None,     # [B] or [B,1]
        sample: bool = True,                     # True: 采样 s；False: 用期望
    ) -> Dict[str, torch.Tensor]:

        h_t = self.encode_history(y_hist_std)                 # [B, d_model]
        alpha, pi, c = self._dirichlet_params(h_t, x_t)       # [B,K], [B,K], [B,1]

        out: Dict[str, torch.Tensor] = {"alpha": alpha, "pi": pi, "c": c}

        if total is not None:
            if total.ndim == 1:
                total = total.unsqueeze(-1)                   # [B,1]
            if sample:
                dist = torch.distributions.Dirichlet(alpha)
                s = dist.rsample()                            # [B, K]
            else:
                s = self._dirichlet_mean(alpha)               # [B, K]
            y_hat = total * s                                 # broadcast: [B,1]*[B,K]
            y_hat, _ = self.laa_dirichlet(s, alpha, total)
            out["y_hat"] = y_hat

        return out

    def elbo_loss(
            self,
            out: dict,
            y_t_raw: torch.Tensor,  # [B, K] 观测的向量（未归一化的“份额×总量”）
            x_t_raw: torch.Tensor,  # [B] or [B,1] 该步总量（分母）
            use_dirichlet_nll: bool = True,  # True: 用 Dirichlet NLL；False: 用 MSE 重构项
            recon_logvar: Optional[torch.Tensor] = None,  # 若用 MSE，可传入 logvar 或用 self.recon_logvar
    ):
        """
        新版 loss（无变分 KL）：
          - Dirichlet NLL:   s_obs = y / total,  loss = -log Dir(alpha | s_obs)
          - 或 MSE 重构：     loss = (y - E[y])^2 / var
        说明：
          你的新网络只有 alpha=c*pi（单头），不再有 posterior/prior 的 alpha_q/alpha_p。
          因此 ELBO 里不再包含 KL(q||p)，直接用观测的 Dirichlet 对数似然即可。
        """
        alpha = out["alpha"]  # [B, K]
        eps = 1e-8

        x_t = x_t_raw[:, :1]  # [B,1]

        # 观测的份额（落在单纯形上）
        s_obs = y_t_raw / (x_t + eps)  # [B, K]
        s_obs = s_obs.clamp_min(eps)
        s_obs = s_obs / s_obs.sum(dim=-1, keepdim=True)  # 避免数值漂移

        if use_dirichlet_nll:
            # 负对数似然（平均）
            dist = torch.distributions.Dirichlet(alpha)
            nll = -dist.log_prob(s_obs)  # [B]
            loss = nll.mean()
            parts = {"nll": nll.mean().detach()}
        else:
            # 用期望重构的 MSE（与旧版风格一致，可选）
            pi = alpha / alpha.sum(dim=-1, keepdim=True)  # E[s] for Dir(alpha)
            y_hat_mean = x_t * pi

            if recon_logvar is None:
                # 若类里有 self.recon_logvar 就用它；否则当成 0（单位方差）
                var_log = getattr(self, "recon_logvar", torch.tensor(0., device=y_t_raw.device))
            else:
                var_log = recon_logvar.to(y_t_raw.device)

            var = torch.exp(var_log) + eps
            recon = ((y_t_raw - y_hat_mean) ** 2 / var).mean() + var_log
            loss = recon
            parts = {"recon": recon.detach()}

        return loss, parts

    @torch.no_grad()
    @torch.no_grad()
    def forecast(
            self,
            y_hist_raw: torch.Tensor,  # [B, T_hist, K]
            x_t_raw: torch.Tensor,  # [B] or [B,1] 该步总量
            x_t_exo: Optional[torch.Tensor] = None,  # [B, d_x] 外生（若没有传 None）
            n_samples: int = 10000,
            use_mean: bool = True,  # True: 用期望；False: 采样
    ):
        """
        返回：
          - y_mean: [K]
          - y_q05 : [K]
          - y_q95 : [K]
          - 以及 alpha/pi/c（便于诊断）
        说明：
          若设备为 MPS，Dirichlet 采样放到 CPU 再搬回。
        """
        device = y_hist_raw.device
        B, T, K = y_hist_raw.shape
        assert K == self.K, f"expect K={self.K}, got {K}"

        # ---------- 标准化（若提供 scaler；否则直接用原值） ----------
        if getattr(self, "sy", None) is not None:
            y_hist_std = torch.from_numpy(
                self.sy.transform(
                    y_hist_raw.detach().cpu().numpy().reshape(-1, K)
                ).reshape(B, T, K)
            ).to(device=device, dtype=y_hist_raw.dtype)
        else:
            y_hist_std = y_hist_raw

        if getattr(self, "sx", None) is not None and x_t_exo is not None:
            x_t_std = torch.from_numpy(
                self.sx.transform(x_t_exo.detach().cpu().numpy())
            ).to(device=device, dtype=x_t_exo.dtype)
        else:
            x_t_std = x_t_exo

        # ---------- h_t & α=c·π ----------
        h_t = self.encode_history(y_hist_std)  # [B, d_model]
        alpha, pi, c = self._dirichlet_params(h_t, x_t_std)  # [B,K],[B,K],[B,1]

        # ---------- 采样 or 期望 ----------
        if x_t_raw.ndim == 1:
            total = x_t_raw [:, :1].unsqueeze(-1)  # [B,1]
        else:
            total = x_t_raw [:, :1]  # [B,1]

        if use_mean:
            s = alpha / alpha.sum(dim=-1, keepdim=True)  # E[s]
            y = total * s  # [B,K]
            y_mean = y.mean(dim=0)
            # 简单用对称带（可选改采样法）
            var = (s * (1 - s)) / (alpha.sum(dim=-1, keepdim=True) + 1.0)
            # 这里省略解析 CI；常见做法还是采样更稳。
            y_mean = self.laa_dirichlet(s, alpha, total)
            y_q05, y_q95 = y_mean, y_mean
        else:
            # 采样
            if alpha.device.type == "mps":
                alpha_cpu = alpha.detach().cpu()
                s_samples_cpu = torch.distributions.Dirichlet(alpha_cpu).sample((n_samples,))  # [S,B,K]
                s_samples = s_samples_cpu.to(device)
            else:
                s_samples = torch.distributions.Dirichlet(alpha).sample((n_samples,))  # [S,B,K]

            s_mean = s_samples.mean(dim=0)
            if total.dim() == 1:
                total_b1 = total.unsqueeze(-1)  # [B,1]
            else:
                total_b1 = total[:, :1]  # [B,1]
            y_mean, _ = self.laa_dirichlet(s_mean, alpha, total_b1)

            y_samples = total.unsqueeze(0) * s_samples  # [S,B,1]*[S,B,K] -> [S,B,K]
            # y_mean = y_samples.mean(dim=0)  # [B,K]
            y_q05 = y_samples.quantile(0.05, dim=0)  # [B,K]
            y_q95 = y_samples.quantile(0.95, dim=0)  # [B,K]

        # 为了和你原接口一致，返回 batch 的统计（B>1 时你可自己再汇总）
        return {
            "alpha": alpha, "pi": pi, "c": c,
            "y_mean": y_mean, "y_q05": y_q05, "y_q95": y_q95,
        }

    def extract_XY(self, df_wide):
        y_cols = sorted([c for c in df_wide.columns if c.startswith("y_")],
                        key=lambda s: int(s.split("_")[1]))
        x_cols = ["x_0"] + sorted([c for c in df_wide.columns if c.startswith("x_") and c != "x_0"],
                                  key=lambda s: int(s.split("_")[1]))
        X = df_wide[x_cols]
        Y = df_wide[y_cols].to_numpy(dtype=np.float32)

        F_total = X.shape[1] - 1
        K = F_total // 2 if F_total > 0 else 0
        harmonic_orders = sum(([k, k] for k in range(1, K + 1)), [])
        return X, Y, y_cols

    def make_seq(self, X, Y, T=32, stride=1):
        N = len(X)
        idx = list(range(0, N - T + 1, stride))
        X_seq = np.stack([X[i:i+T] for i in idx], axis=0)
        Y_seq = np.stack([Y[i:i+T] for i in idx], axis=0)
        return X_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, X_seq, Y_seq):
            self.X = torch.from_numpy(X_seq).float()
            self.Y = torch.from_numpy(Y_seq).float()
        def __len__(self):  return self.X.shape[0]
        def __getitem__(self, i):  return self.X[i], self.Y[i]

    def get_dataloader(self, df, mode = 'vector'):
        df['ds'] = pd.to_datetime(df['ds'])

        df = simple_dst_fix(df)

        df['day'] = df['ds'].dt.date
        df["hour"] = df["ds"].dt.hour
        df_wide = (
            df
            .pivot(index="day", columns="hour", values="y")
            .add_prefix("y_")
            .rename_axis(None, axis=1)
            .reset_index()
        )
        df_wide['x_0'] = df_wide.filter(like="y_").sum(axis=1)

        df_wide, _ = append_fourier_features(df_wide, K=5,
                                             period_days=7.0, mode=mode)

        X, Y, y_cols = self.extract_XY(df_wide)
        X_train, Y_train = X[:-1], Y[:-1]
        X_test,  Y_test  = X[-1:], Y[-1:]

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        X_std = sx.transform(X_train)
        Y_std = sy.transform(Y_train)

        X_seq, Y_seq = self.make_seq(X_std, Y_std, T=32, stride=1)
        ds = self.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        self.sx, self.sy = sx, sy

        return (loader, sx, sy, X_train, Y_train, X_test, Y_test)


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class RNN_train:
    def __init__(self, model, training_config, train_sample = True, mode = 'vector'):
        self.model = model
        self.cfg = training_config
        self.model.to(self.cfg.device)
        self.train_sample = train_sample
        self.mode = mode

    def __call__(self, df):
        # ── 取数据：内部 helper 会把 scaler 存到 model.sx / model.sy ─────────
        loader, sx, sy, X_train, Y_train, X_test, Y_test = self.model.get_dataloader(df, mode = self.mode)

        # ── 优化器 ───────────────────────────────────────────────────────────────
        opt = optim.Adam(
            self.model.parameters(),
            lr=getattr(self.cfg, "lr", 5e-4),
            betas=getattr(self.cfg, "betas", (0.9, 0.999)),
            weight_decay=getattr(self.cfg, "weight_decay", 0.0),
        )
        max_norm     = getattr(self.cfg, "max_grad_norm", 1.0)
        T_hist       = getattr(self.cfg, "T_hist", 32)
        pred_samples = getattr(self.cfg, "pred_samples", 10000)

        # ── 小工具：反标准化一个 batch（二维向量 [B,H]）────────────────────────
        def _inv_std_vec(scaler, t2d: torch.Tensor) -> torch.Tensor:
            # t2d: [B, H]
            arr = t2d.detach().cpu().numpy()
            inv = scaler.inverse_transform(arr)
            return torch.from_numpy(inv).to(t2d.device, dtype=t2d.dtype)

        for epoch in range(self.cfg.n_epochs):
            self.model.train()
            total_loss = 0.0
            total_nll  = 0.0
            n_batches  = 0

            for X_seq_std, Y_seq_std in loader:
                X_seq_std = X_seq_std.to(self.cfg.device)  # [B, T, d_x]（通常 d_x=1，总量）
                Y_seq_std = Y_seq_std.to(self.cfg.device)  # [B, T, K]

                B, T, H = Y_seq_std.shape
                assert T == T_hist, "test range not match"

                # ── 切分历史/当前步（注意：此时仍是标准化的）──────────────────
                y_hist_std = Y_seq_std[:, :-1, :]          # [B, T-1, K]
                y_t_std    = Y_seq_std[:, -1, :]           # [B, K]
                x_t_std    = X_seq_std[:, -1, :]           # [B, d_x]（若 d_x=1 则 [B,1]）

                # ── 反标准化得到“原始量纲”的 y_t_raw / x_t_raw（用于 NLL）─────
                y_t_raw = _inv_std_vec(self.model.sy, y_t_std)   # [B, K]
                x_t_raw = _inv_std_vec(self.model.sx, x_t_std)   # [B, d_x]→通常 [B,1]

                # ── 前向：给网络标准化后的历史（和可选外生），并把 total 传 raw ──
                opt.zero_grad(set_to_none=True)
                # 你的新网络 forward(y_hist_std, x_t=..., total=..., sample=False)
                out = self.model(y_hist_std, x_t=x_t_std, total= x_t_raw[:, :1], sample=self.train_sample)

                # ── 损失：默认用 Dirichlet NLL（对 s_obs = y/x）────────────────
                loss, logs = self.model.elbo_loss(
                    out,
                    y_t_raw=y_t_raw,
                    x_t_raw=x_t_raw.squeeze(-1) if x_t_raw.ndim == 2 and x_t_raw.shape[1] == 1 else x_t_raw,
                    use_dirichlet_nll=True
                )

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                opt.step()

                total_loss += float(loss.detach())
                total_nll  += float(logs.get("nll", loss).detach())
                n_batches  += 1

            print(f"[Epoch {epoch+1:03d}] loss={total_loss/n_batches:.4f} nll={total_nll/n_batches:.4f}")

        # ── 预测一个点（与你原逻辑一致）：这里传“原始量纲”的历史与 total ────────
        y_hist_raw = torch.from_numpy(Y_train[-T_hist:, :][None, ...]).to(self.cfg.device)  # [1,T,K]
        print(X_test.iloc[-1:, :].to_numpy(np.float32).shape)
        x_t_raw = torch.from_numpy(X_test.iloc[-1:, :].to_numpy(np.float32)).to(self.cfg.device)                   # [1,d_x]

        pred = self.predict_one(y_hist_raw, x_t_raw, n_samples=pred_samples)
        Y_pred = pred["y_mean"].cpu()

        return {
            "sx": sx, "sy": sy,
            "X_train": X_train, "Y_train": Y_train,
            "X_test": X_test,   "Y_test": torch.from_numpy(Y_test).cpu(),
            "Y_pred": Y_pred
        }

    @torch.no_grad()
    def predict_one(self, y_hist_raw, x_t_raw, n_samples=10000):
        self.model.eval()
        # 新版 forecast(y_hist_raw, x_t_raw, x_t_exo=None, n_samples=...)
        out = self.model.forecast(
            y_hist_raw.to(self.cfg.device),
            x_t_raw.to(self.cfg.device),
            x_t_exo=None,
            n_samples=n_samples,
            use_mean=False
        )
        return {
            "y_mean": out["y_mean"].cpu(),
            "y_q05":  out["y_q05"].cpu(),
            "y_q95":  out["y_q95"].cpu()
        }
