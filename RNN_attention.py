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


class RNN_attention(nn.Module):

    def __init__(
            self,
            latent_dim=24,
            d_model=128,
            nhead=4,
            activation="relu",
            learn_z0=True,
            dropout=0.0,
            H=24,
            use_gate=True,
            rnn_type="rnn"
    ):
        super().__init__()

        self.H = H
        self.latent_dim = latent_dim

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

    def forward(self, x_cont, z0=None):

        B, T, C = x_cont.shape
        assert C == 1, f"expect input_dim=1 (only x0), got {C}"

        if z0 is None and self.learn_z0:
            h0 = self.z0.expand(-1, B, -1).contiguous()
        else:
            h0 = z0

        h, _ = self.rnn(x_cont, h0)

        z = self._feature_attend(h)

        o = z @ self.A.T + self.c

        return o, z

    @staticmethod
    def extract_XY(df_wide):
        y_cols = sorted([c for c in df_wide.columns if c.startswith("y_")],
                        key=lambda s: int(s.split("_")[1]))
        X = df_wide[["x_0"]]
        Y = df_wide[y_cols].to_numpy(dtype=np.float32)

        return X, Y, y_cols

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

    def get_dataloader(self, df):

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

        X, Y, _ = self.extract_XY(df_wide)

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

        return (loader, sx, sy, X_train, Y_train, X_test, Y_test)


@dataclass
class training_config:
    n_epochs: int = 25
    device: torch.device = torch.device("cpu")
    T_hist: int = 32
    lr: float = 5e-4


class RNN_train_attention:
    def __init__(self, model, training_config):
        self.model = model
        self.training_config = training_config
        self.model.to(self.training_config.device)

    def __call__(self, df):
        loader, sx, sy, X_train, Y_train, X_test, Y_test = self.model.get_dataloader(df)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.training_config.lr)

        for ep in range(self.training_config.n_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.training_config.device), yb.to(self.training_config.device)
                opt.zero_grad()
                o, _ = self.model(xb)
                loss = F.mse_loss(o, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
            print(f"epoch {ep + 1} loss: {float(loss):.4f}")

        y_pred = self.model.forecast_knownX(X_hist=X_train, X_future=X_test, T=32, sx=sx, sy=sy)
        return list(y_pred.flatten()), list(Y_test.flatten())