import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from torch.backends.cudnn import deterministic
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Optional, Literal, Tuple, List, Dict
from dataclasses import dataclass

from scipy.stats import norm


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
                            start_idx=None,
                            start_day = None):
    if K <= 0:
        return df_wide, 0

    existing = [c for c in df_wide.columns if c.startswith("x_") and c != "x_0"]
    if start_idx is None:
        start_idx = 1 + len(existing)
    t0 = pd.to_datetime(df_wide["day"]).astype("int64") // 86_400_000_000_000
    if start_day is None:
        start_day = t0.min()
    t_day = (t0 - start_day).astype(float).to_numpy()
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

    def __init__(
            self,
            cont_dim=0,
            fourier_dim=0,
            xf_mode="vector",
            latent_dim=24,
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

        self.H = H
        self.latent_dim = latent_dim
        self.F = fourier_dim
        self.mode = xf_mode
        self.nonneg_U0 = nonneg_U0
        self.cont_dim = cont_dim

        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=1,
                hidden_size=latent_dim,
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

        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(1, 1, latent_dim))

        if self.mode == "vector":

            self.Uf = nn.Parameter(torch.randn(latent_dim, self.F) * 0.01)
        else:

            self.Uf = nn.Parameter(torch.randn(self.H, self.F) * 0.01)

            if latent_dim != self.H:
                self.fourier_proj = nn.Linear(self.H, latent_dim)
            else:
                self.fourier_proj = None

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

        self.A = nn.Parameter(torch.randn(self.H, latent_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(self.H))

    def _feature_attend(self, z):

        B, T, L = z.shape
        z_flat = z.reshape(B * T, L)

        tokens = z_flat.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.ffn(tokens))
        delta = self.out_proj(tokens).squeeze(-1)

        delta = delta.reshape(B, T, L)

        if self.use_gate:
            return z + self.gate(z) * delta
        else:
            return z + delta

    def reg_loss(self, lambda0=0.0, lambdaf=0.0, harmonic_orders=None):

        loss = torch.tensor(0.0, device=self.Uf.device)

        if lambdaf > 0:
            if harmonic_orders is None:
                loss = loss + lambdaf * (self.Uf ** 2).sum()
            else:
                w = torch.as_tensor(harmonic_orders, dtype=self.Uf.dtype, device=self.Uf.device)
                loss = loss + lambdaf * ((self.Uf ** 2) * (w ** 2).unsqueeze(0)).sum()

        return loss

    def forward(self, x_cont, z0=None):

        B, T, C = x_cont.shape
        assert C == self.cont_dim, f"expect cont_dim={self.cont_dim}, got {C}"

        x0 = x_cont[:, :, :1]
        xf = x_cont[:, :, 1:]

        if z0 is None and self.learn_z0:
            h0 = self.z0.expand(-1, B, -1).contiguous()
        else:
            h0 = z0

        h, _ = self.rnn(x0, h0)

        if self.mode == "vector":

            fourier_contrib = xf @ self.Uf.T
        else:
            xf_mat = xf.view(B, T, self.H, self.F)
            fourier_h = (xf_mat * self.Uf.unsqueeze(0).unsqueeze(0)).sum(-1)

            if self.fourier_proj is not None:
                fourier_contrib = self.fourier_proj(fourier_h)
            else:
                fourier_contrib = fourier_h

        z = h + fourier_contrib

        z = self._feature_attend(z)

        o = z @ self.A.T + self.c

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
        self.eval()
        dev = next(self.parameters()).device

        hist = np.asarray(X_hist[-T:], dtype=np.float32)
        fut = np.asarray(X_future, dtype=np.float32)

        if sx is not None:
            hist = sx.transform(hist).astype(np.float32)
            fut = sx.transform(fut).astype(np.float32)

        x_in = np.concatenate([hist, fut], axis=0)[None, ...]
        x_in = torch.from_numpy(x_in).to(dev)

        o_all, _ = self(x_in)
        y_std = o_all[0, -len(X_future):, :].cpu().numpy()

        return sy.inverse_transform(y_std) if sy is not None else y_std

    def get_dataloader(self, df, fourier_config, test=False, start_day=None):

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
                                             period_days=fourier_config.P_WEEK, mode=fourier_config.mode,
                                             start_day=start_day)
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_monthly,
                                             period_days=fourier_config.P_MONTH, mode=fourier_config.mode,
                                             start_day=start_day)
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_yearly,
                                             period_days=fourier_config.P_yearly, mode=fourier_config.mode,
                                             start_day=start_day)

        X, Y, _, _, _ = self.extract_XY(df_wide)

        if not test:
            X_np = np.asarray(X, dtype=np.float32)
            Y_np = np.asarray(Y, dtype=np.float32)

            sx = StandardScaler().fit(X_np)
            sy = StandardScaler().fit(Y_np)

            Xs_tr = sx.transform(X_np).astype(np.float32)
            Ys_tr = sy.transform(Y_np).astype(np.float32)

            X_seq, Y_seq = self.make_seq(Xs_tr, Ys_tr, T=32, stride=1)
            loader = DataLoader(self.XYSeqDataset(X_seq, Y_seq), batch_size=64, shuffle=True)
            F_total = X.shape[1] - 1

            t0 = pd.to_datetime(df_wide["day"]).astype("int64") // 86_400_000_000_000

            return (loader, sx, sy, X, Y, F_total, t0.min())

        else:

            return X, Y


def generate_prediction_intervals(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        epsilon_mean: np.ndarray,
        epsilon_cov: np.ndarray,
        confidence: float = 0.95,
        n_samples: int = 50000) -> Dict[str, np.ndarray]:

    y_pred = np.asarray(y_pred)

    epsilon_samples = np.random.multivariate_normal(
        np.zeros_like(epsilon_mean),
        epsilon_cov,
        size=n_samples
    )

    if y_pred.ndim == 1:
        y_samples = y_pred[np.newaxis, :] + epsilon_samples
    else:
        n_days = y_pred.shape[0]
        y_samples = np.zeros((n_samples, n_days, 24))
        for d in range(n_days):
            y_samples[:, d, :] = y_pred[d] + epsilon_samples

    alpha = 1 - confidence
    lower = np.quantile(y_samples, alpha / 2, axis=0)
    upper = np.quantile(y_samples, 1 - alpha / 2, axis=0)
    point = y_pred + epsilon_mean

    result = {
        'point': point,
        'lower': lower,
        'upper': upper,
        'width': upper - lower,
        'samples': y_samples
    }

    if y_true is not None:
        y_true = np.asarray(y_true)

        if y_pred.ndim == 1:
            p_values = np.zeros(24)

            for h in range(24):
                p_values[h] = np.mean(y_samples[:, h] <= y_true[h])

            result['p_values'] = p_values

        else:
            n_days = y_pred.shape[0]
            p_values = np.zeros((n_days, 24))

            for d in range(n_days):
                for h in range(24):
                    p_values[d, h] = np.mean(y_samples[:, d, h] <= y_true[d, h])

            result['p_values'] = p_values

    return result


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
        self.start_day = None

    def __call__(self, df_train):
        loader, sx, sy, X_train, Y_train, F_total, start_day = self.model.get_dataloader(df_train, self.fourier_conf)
        self.start_day = start_day
        self.sx = sx
        self.sy = sy
        self.X_train = X_train

        if self.model.mode == "vector":
            K_est = F_total // 2
        elif self.model.mode == "matrix":
            K_est = (F_total // (2 * 24))
        else:
            raise ValueError("mode must be 'vector' or 'matrix'")

        harmonic_orders = sum(([k, k] for k in range(1, K_est + 1)), [])
        device = self.training_config.device
        opt = torch.optim.Adam(self.model.parameters(), lr=self.training_config.lr)

        # Training loop
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

        self.model.eval()

        train_residuals_matrix = []
        T = self.training_config.T_hist

        with torch.no_grad():
            # Start from index T_hist, predict one day at a time
            for i in range(T, len(X_train)):
                X_hist = X_train[i - T:i]
                X_fut = X_train[i:i + 1]

                pred = self.model.forecast_knownX(X_hist=X_hist, X_future=X_fut, T=T, sx=sx, sy=sy)
                y_true = Y_train[i]

                residual = y_true - pred.flatten()
                train_residuals_matrix.append(residual)

        train_residuals_matrix = np.array(train_residuals_matrix)

        self.epsilon_mean = np.mean(train_residuals_matrix, axis=0)
        self.epsilon_cov = np.cov(train_residuals_matrix, rowvar=False)
        self.epsilon_std = np.sqrt(np.diag(self.epsilon_cov))

    def forcaste(self, df_test, deterministic = False):
        X_test, Y_test = self.model.get_dataloader(df_test, self.fourier_conf, test=True, start_day = self.start_day)
        T = self.training_config.T_hist
        with torch.no_grad():
            y_test_pred = self.model.forecast_knownX(X_hist=self.X_train, X_future=X_test, T=T, sx=self.sx, sy=self.sy)

        if deterministic:
            return y_test_pred.flatten(), Y_test.flatten()

        intervals = generate_prediction_intervals(
            y_pred=y_test_pred.flatten(),
            y_true=Y_test.flatten(),
            epsilon_mean=self.epsilon_mean,
            epsilon_cov=self.epsilon_cov,
            confidence=0.95,
            n_samples=50000
        )

        return {
            'test_pred': y_test_pred.flatten(),
            'test_true': Y_test.flatten(),
            'y_pred_lower':intervals['lower'],
            'y_pred_upper':intervals['upper'],
            'p_values': intervals['p_values'].flatten(),
        }
    def forecate(self, df_test, deterministic = False):
        X_test, Y_test = self.model.get_dataloader(df_test, self.fourier_conf, test=True, start_day = self.start_day)
        T = self.training_config.T_hist
        with torch.no_grad():
            y_test_pred = self.model.forecast_knownX(X_hist=self.X_train, X_future=X_test, T=T, sx=self.sx, sy=self.sy)

        if deterministic:
            return y_test_pred.flatten(), Y_test.flatten()

        intervals = generate_prediction_intervals(
            y_pred=y_test_pred.flatten(),
            y_true=Y_test.flatten(),
            epsilon_mean=self.epsilon_mean,
            epsilon_cov=self.epsilon_cov,
            confidence=0.95,
            n_samples=50000
        )

        return {
            'test_pred': y_test_pred.flatten(),
            'test_true': Y_test.flatten(),
            'y_pred_lower':intervals['lower'],
            'y_pred_upper':intervals['upper'],
            'p_values': intervals['p_values'].flatten(),
        }
