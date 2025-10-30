import torch
import torch.nn as nn
import torch.nn.functional as F


def build_directional_masks(H: int, device=None, include_self: bool = True):


    if device is None:
        device = torch.device("cpu")
    zero = torch.zeros((H, H), device=device)
    neginf = torch.full((H, H), float("-inf"), device=device)

    tril = torch.tril(torch.ones(H, H, device=device), diagonal=0 if include_self else -1)
    triu = torch.triu(torch.ones(H, H, device=device), diagonal=0 if include_self else 1)

    past_mask = torch.where(tril.bool(), zero, neginf)
    future_mask = torch.where(triu.bool(), zero, neginf)

    return past_mask, future_mask


class RelativeDirectionalBias(nn.Module):

    def __init__(self, H: int, separate_past_future: bool = True):
        super().__init__()
        self.H = H
        self.num_rel = 2 * H - 1
        if separate_past_future:
            self.bias_past = nn.Parameter(torch.zeros(self.num_rel))
            self.bias_future = nn.Parameter(torch.zeros(self.num_rel))
        else:
            self.bias_shared = nn.Parameter(torch.zeros(self.num_rel))
        self.separate = separate_past_future

    def forward(self, H: int, device=None, for_future: bool = False):

        if device is None:
            device = torch.device("cpu")
        idx_i = torch.arange(H, device=device).unsqueeze(1)
        idx_j = torch.arange(H, device=device).unsqueeze(0)
        rel = idx_j - idx_i
        rel_shift = rel + (H - 1)

        if self.separate:
            bias_vec = self.bias_future if for_future else self.bias_past
        else:
            bias_vec = self.bias_shared
        B = bias_vec[rel_shift]
        return B


class AsymmetricFeatureAttention(nn.Module):

    def __init__(self,
                 H: int = 24,
                 d_model: int = 128,
                 nhead: int = 4,
                 dropout: float = 0.0,
                 use_dir_bias: bool = True,
                 separate_dir_bias: bool = True,
                 use_softmax_gating: bool = True):
        super().__init__()
        self.H = H
        self.d_model = d_model
        self.nhead = nhead
        self.use_dir_bias = use_dir_bias
        self.use_softmax_gating = use_softmax_gating

        self.mha_past = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mha_future = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.out_proj_past = nn.Linear(d_model, 1)
        self.out_proj_future = nn.Linear(d_model, 1)

        if use_softmax_gating:
            self.alpha_logits = nn.Parameter(torch.zeros(2))
        else:
            self.alpha_past = nn.Parameter(torch.tensor(0.0))
            self.alpha_future = nn.Parameter(torch.tensor(0.0))

        if use_dir_bias:
            self.dir_bias = RelativeDirectionalBias(H, separate_past_future=separate_dir_bias)
        else:
            self.dir_bias = None

        self.register_buffer("past_mask", None, persistent=False)
        self.register_buffer("future_mask", None, persistent=False)

    def _ensure_masks(self, device):
        if self.past_mask is None or self.future_mask is None or self.past_mask.device != device:
            pm, fm = build_directional_masks(self.H, device=device, include_self=True)
            self.past_mask = pm
            self.future_mask = fm

    def forward(self, z: torch.Tensor, feat_embed: torch.Tensor):

        B, H = z.shape
        assert H == self.H, f"H mismatch: got {H}, expected {self.H}"
        device = z.device
        self._ensure_masks(device)
        tokens = z.unsqueeze(-1) * feat_embed.unsqueeze(0)

        if self.dir_bias is not None:
            B_past = self.dir_bias(H=self.H, device=device, for_future=False)
            B_future = self.dir_bias(H=self.H, device=device, for_future=True)
            attn_mask_past = torch.where(self.past_mask == 0, B_past, self.past_mask)
            attn_mask_future = torch.where(self.future_mask == 0, B_future, self.future_mask)
        else:
            attn_mask_past = self.past_mask
            attn_mask_future = self.future_mask

        past_out, past_w = self.mha_past(tokens, tokens, tokens, attn_mask=attn_mask_past)
        future_out, future_w = self.mha_future(tokens, tokens, tokens, attn_mask=attn_mask_future)

        tokens_past = self.ln1(tokens + past_out)
        tokens_future = self.ln1(tokens + future_out)

        tokens_past = self.ln2(tokens_past + self.ffn(tokens_past))
        tokens_future = self.ln2(tokens_future + self.ffn(tokens_future))

        delta_past = self.out_proj_past(tokens_past).squeeze(-1)
        delta_future = self.out_proj_future(tokens_future).squeeze(-1)

        if self.use_softmax_gating:
            alphas = F.softmax(self.alpha_logits, dim=0)
            a_past, a_future = alphas[0], alphas[1]
        else:
            a_past = torch.sigmoid(self.alpha_past)
            a_future = torch.sigmoid(self.alpha_future)

        delta = a_past * delta_past + a_future * delta_future

        aux = {
            "alpha_past": a_past.detach().item(),
            "alpha_future": a_future.detach().item(),
            "attn_weights_past": past_w.detach(),
            "attn_weights_future": future_w.detach(),
        }
        return delta, aux
