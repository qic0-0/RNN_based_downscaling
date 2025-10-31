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


class RNN_fourier(nn.Module):
    """
    RNN with output-only Fourier features.

    Architecture:
    1. RNN processes base dynamics (x0 only) in latent space
    2. Fourier features are added to latent representation
    3. Feature attention refines the combined representation
    4. Final linear projection to output (24 hours)
    """

    def __init__(
            self,
            cont_dim=0,
            fourier_dim=0,
            xf_mode="vector",
            latent_dim=24,  # New parameter: dimension of RNN hidden state
            d_model=128,
            nhead=4,
            activation="relu",
            learn_z0=True,
            dropout=0.0,
            H=24,
            use_gate=True,
            nonneg_U0=False,
            rnn_type="rnn"
    ):
        super().__init__()
        assert xf_mode in ("vector", "matrix")
        if xf_mode == "vector":
            assert cont_dim == 1 + fourier_dim, "vector mode cont_dim should be 1+F"
        else:
            assert cont_dim == 1 + 24 * fourier_dim, "matrix mode cont_dim should be 1+24*F"

        self.H = H  # Output dimension (24 hours)
        self.latent_dim = latent_dim  # RNN hidden state dimension
        self.F = fourier_dim
        self.mode = xf_mode
        self.nonneg_U0 = nonneg_U0
        self.cont_dim = cont_dim

        # RNN for base dynamics (only processes x0)
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=1,  # Only x0 (daily total)
                hidden_size=latent_dim,  # Can be different from H
                num_layers=1,
                batch_first=True,
                nonlinearity='relu' if activation == 'relu' else 'tanh'
            )
        else:
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=latent_dim,
                num_layers=1,
                batch_first=True
            )

        # Optional: learnable initial hidden state
        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(1, 1, latent_dim))  # (num_layers, batch, latent_dim)

        # Fourier feature projection to latent space
        if self.mode == "vector":
            # Vector mode: same Fourier features for all hours
            # Project Fourier features to latent space
            self.Uf = nn.Parameter(torch.randn(latent_dim, self.F) * 0.01)
        else:
            # Matrix mode: different Fourier features for each hour
            # In this case, we need to match the output dimension H
            # We'll project Fourier to H-dimensional space, then to latent
            self.Uf = nn.Parameter(torch.randn(self.H, self.F) * 0.01)
            # If latent_dim != H, add projection layer
            if latent_dim != self.H:
                self.fourier_proj = nn.Linear(self.H, latent_dim)
            else:
                self.fourier_proj = None

        # Feature attention mechanism (operates on latent space)
        self.feat_embed = nn.Parameter(torch.randn(latent_dim, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Sigmoid())

        # Final output projection from latent space to H hours
        self.A = nn.Parameter(torch.randn(self.H, latent_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(self.H))

    def _feature_attend(self, z):
        """
        Apply self-attention over the latent features.
        z: (B, T, latent_dim)

        We process each timestep independently, applying attention over the latent_dim features.
        """
        B, T, L = z.shape  # L = latent_dim

        # Reshape to (B*T, latent_dim) to process all timesteps together
        z_flat = z.reshape(B * T, L)

        # Create feature embeddings: (B*T, latent_dim, d_model)
        tokens = z_flat.unsqueeze(-1) * self.feat_embed.unsqueeze(0)

        # Apply attention over latent_dim dimension
        # Input shape: (B*T, latent_dim, d_model)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.ffn(tokens))
        delta = self.out_proj(tokens).squeeze(-1)  # (B*T, latent_dim)

        # Reshape back to (B, T, latent_dim)
        delta = delta.reshape(B, T, L)

        if self.use_gate:
            return z + self.gate(z) * delta
        else:
            return z + delta

    def reg_loss(self, lambda0=0.0, lambdaf=0.0, harmonic_orders=None):
        """
        Regularization loss for Fourier coefficients.
        Penalizes higher harmonics more heavily.
        """
        loss = torch.tensor(0.0, device=self.Uf.device)

        if lambdaf > 0:
            if harmonic_orders is None:
                loss = loss + lambdaf * (self.Uf ** 2).sum()
            else:
                w = torch.as_tensor(harmonic_orders, dtype=self.Uf.dtype, device=self.Uf.device)
                # Harmonic orders: [1,1,2,2,3,3,...] for [sin1,cos1,sin2,cos2,...]
                loss = loss + lambdaf * ((self.Uf ** 2) * (w ** 2).unsqueeze(0)).sum()

        return loss

    def forward(self, x_cont, z0=None):
        """
        Forward pass.

        Args:
            x_cont: (B, T, cont_dim) where cont_dim = 1 + F (vector) or 1 + 24*F (matrix)
            z0: Optional initial hidden state (num_layers, B, latent_dim)

        Returns:
            O: (B, T, H) - output predictions for H hours (typically 24)
            Z: (B, T, latent_dim) - final hidden states in latent space
        """
        B, T, C = x_cont.shape
        assert C == self.cont_dim, f"expect cont_dim={self.cont_dim}, got {C}"

        # Split input into base (x0) and Fourier (xf)
        x0 = x_cont[:, :, :1]  # (B, T, 1) - daily totals
        xf = x_cont[:, :, 1:]  # (B, T, F) or (B, T, 24*F)

        # RNN processes only base dynamics in latent space
        if z0 is None and self.learn_z0:
            h0 = self.z0.expand(-1, B, -1).contiguous()  # (1, B, latent_dim)
        else:
            h0 = z0

        h, _ = self.rnn(x0, h0)  # h: (B, T, latent_dim)

        # Add Fourier contribution to latent space
        if self.mode == "vector":
            # Vector mode: xf is (B, T, F)
            # Same Fourier pattern applied to latent space
            fourier_contrib = xf @ self.Uf.T  # (B, T, F) @ (F, latent_dim) -> (B, T, latent_dim)
        else:
            # Matrix mode: xf is (B, T, 24*F)
            # Reshape to (B, T, 24, F) - different Fourier for each hour
            xf_mat = xf.view(B, T, self.H, self.F)  # (B, T, 24, F)
            # Each hour h gets: sum_f(xf[h,f] * Uf[h,f])
            fourier_h = (xf_mat * self.Uf.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B, T, 24)

            # Project to latent space if dimensions don't match
            if self.fourier_proj is not None:
                fourier_contrib = self.fourier_proj(fourier_h)  # (B, T, 24) -> (B, T, latent_dim)
            else:
                fourier_contrib = fourier_h  # (B, T, 24) when latent_dim == 24

        # Combine RNN output with Fourier features in latent space
        z = h + fourier_contrib  # (B, T, latent_dim)

        # Apply feature attention in latent space
        z = self._feature_attend(z)  # (B, T, latent_dim)

        # Final output projection from latent space to H hours
        o = z @ self.A.T + self.c  # (B, T, latent_dim) @ (latent_dim, H) -> (B, T, H)

        return o, z

    @staticmethod
    def extract_XY(df_wide):
        y_cols = sorted([c for c in df_wide.columns if c.startswith("y_")],
                        key=lambda s: int(s.split("_")[1]))
        x_cols = ["x_0"] + sorted([c for c in df_wide.columns if c.startswith("x_") and c != "x_0"],
                                  key=lambda s: int(s.split("_")[1]))
        X = df_wide[x_cols]
        Y = df_wide[y_cols].to_numpy(dtype=np.float32)

        F_total = X.shape[1] - 1
        K = F_total // 2 if F_total > 0 else 0
        harmonic_orders = sum(([k, k] for k in range(1, K + 1)), [])
        return X, Y, y_cols, x_cols, harmonic_orders

    @staticmethod
    def make_seq(X, Y, T=32, stride=1):
        N = len(X)
        idx = list(range(0, N - T + 1, stride))
        X_seq = np.stack([X[i:i + T] for i in idx], axis=0).astype(np.float32)
        Y_seq = np.stack([Y[i:i + T] for i in idx], axis=0).astype(np.float32)
        return X_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, X_seq, Y_seq):
            self.X = torch.from_numpy(X_seq).float()
            self.Y = torch.from_numpy(Y_seq).float()

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return self.X[i], self.Y[i]

    @torch.no_grad()
    def forecast_knownX(self, X_hist, X_future, T, sx=None, sy=None):
        """
        Generate forecast given historical and future features.

        Args:
            X_hist: Historical features (N_hist, cont_dim)
            X_future: Future features (N_future, cont_dim)
            T: Context length
            sx: StandardScaler for X
            sy: StandardScaler for Y

        Returns:
            Forecasted values (N_future, 24)
        """
        self.eval()
        dev = next(self.parameters()).device

        hist = np.asarray(X_hist[-T:], dtype=np.float32)
        fut = np.asarray(X_future, dtype=np.float32)

        if sx is not None:
            hist = sx.transform(hist).astype(np.float32)
            fut = sx.transform(fut).astype(np.float32)

        x_in = np.concatenate([hist, fut], axis=0)[None, ...]  # (1, T+N_future, cont_dim)
        x_in = torch.from_numpy(x_in).to(dev)

        o_all, _ = self(x_in)
        y_std = o_all[0, -len(X_future):, :].cpu().numpy()  # (N_future, 24)

        return sy.inverse_transform(y_std) if sy is not None else y_std

    def get_dataloader(self, df, fourier_config):
        """
        Prepare data loader from raw dataframe.

        Args:
            df: DataFrame with columns ['ds', 'y']
            fourier_config: Configuration for Fourier features

        Returns:
            Tuple of (loader, sx, sy, X_train, Y_train, X_test, Y_test, F_total)
        """
        df['ds'] = pd.to_datetime(df['ds'])

        # Fix missing data and duplicate data and na
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

        # Add Fourier features
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_weekly,
                                             period_days=fourier_config.P_WEEK, mode=fourier_config.mode)
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_monthly,
                                             period_days=fourier_config.P_MONTH, mode=fourier_config.mode)
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_yearly,
                                             period_days=fourier_config.P_yearly, mode=fourier_config.mode)

        X, Y, _, _, _ = self.extract_XY(df_wide)

        X_train, Y_train = X[:-1], Y[:-1]
        X_test, Y_test = X[-1:], Y[-1:]

        X_train_np = np.asarray(X_train, dtype=np.float32)
        Y_train_np = np.asarray(Y_train, dtype=np.float32)
        X_test_np = np.asarray(X_test, dtype=np.float32)

        sx = StandardScaler().fit(X_train_np)
        sy = StandardScaler().fit(Y_train_np)

        Xs_tr = sx.transform(X_train_np).astype(np.float32)
        Ys_tr = sy.transform(Y_train_np).astype(np.float32)
        X_std_test = sx.transform(X_test_np).astype(np.float32)

        X_seq, Y_seq = self.make_seq(Xs_tr, Ys_tr, T=32, stride=1)
        loader = DataLoader(self.XYSeqDataset(X_seq, Y_seq), batch_size=64, shuffle=True)
        F_total = X.shape[1] - 1

        return (loader, sx, sy, X_train, Y_train, X_test, Y_test, F_total)


@dataclass
class training_config:
    n_epochs: int = 25
    device: torch.device = torch.device("cpu")
    T_hist: int = 32
    lr: float = 5e-4
    kl_coeff: float = 1.0
    pred_samples: int = 1000
    lambda0: float = 1e-5
    lambdaf: float = 5e-4


@dataclass
class fourier_config:
    mode: Literal["vector", "matrix"] = "vector"
    K_weekly: int = 0
    K_monthly: int = 0
    K_yearly: int = 0
    P_WEEK: float = 7.0
    P_MONTH: float = 365.25 / 12.0
    P_yearly: float = 365.25


class RNN_train_fourier:
    def __init__(self, model, training_config, fourier_conf):
        self.model = model
        self.training_config = training_config
        self.fourier_conf = fourier_conf
        self.model.to(self.training_config.device)

    def __call__(self, df):
        loader, sx, sy, X_train, Y_train, X_test, Y_test, F_total = self.model.get_dataloader(df, self.fourier_conf)

        if self.model.mode == "vector":
            K_est = F_total // 2
        elif self.model.mode == "matrix":
            K_est = (F_total // (2 * 24))
        else:
            raise ValueError("mode must be 'vector' or 'matrix'")

        harmonic_orders = sum(([k, k] for k in range(1, K_est + 1)), [])
        device = self.training_config.device
        opt = torch.optim.Adam(self.model.parameters(), lr=self.training_config.lr)

        for ep in range(self.training_config.n_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                o, _ = self.model(xb)
                data_loss = F.mse_loss(o, yb)
                prior_loss = self.model.reg_loss(
                    lambda0=self.training_config.lambda0,
                    lambdaf=self.training_config.lambdaf,
                    harmonic_orders=harmonic_orders
                )
                loss = data_loss + prior_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
            print(f"epoch {ep + 1} loss: {float(loss):.4f}")

        y_pred = self.model.forecast_knownX(X_hist=X_train, X_future=X_test, T=32, sx=sx, sy=sy)
        return list(y_pred.flatten()), list(Y_test.flatten())