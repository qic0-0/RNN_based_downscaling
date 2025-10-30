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


class RNN_FeatureAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4, activation="tanh", learn_z0=True, dropout=0.0):
        super().__init__()
        self.h = 24

        self.W = nn.Parameter(torch.randn(self.h, self.h) * 0.01)
        self.U = nn.Parameter(torch.randn(self.h, 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.h))

        self.A = nn.Parameter(torch.randn(24, self.h) * 0.01)
        self.c = nn.Parameter(torch.zeros(24))

        # self attention
        self.feat_embed = nn.Parameter(torch.randn(24, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                          batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(self.h))

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

    def _feature_attend(self, z):
        B = z.shape[0]
        tokens = z.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.ln2(tokens + ffn_out)
        delta = self.out_proj(tokens).squeeze(-1)
        return z + delta

    def forward(self, x, z0=None):
        B, T, _ = x.shape
        if z0 is None:
            if self.learn_z0:
                z_t = self.z0.expand(B, self.h)
            else:
                z_t = torch.zeros(B, self.h, device=x.device)
        else:
            z_t = z0

        Z, O = [], []
        for t in range(T):
            x_t = x[:, t, :]
            z_t = self.act(z_t @ self.W.T + x_t @ self.U.T + self.b)
            z_t = self._feature_attend(z_t)
            Z.append(z_t)
            o_t = z_t @ self.A.T + self.c
            O.append(o_t)

        Z = torch.stack(Z, dim=1)
        O = torch.stack(O, dim=1)
        return O, Z

    def extract_XY(self, df):
        y_cols = sorted([c for c in df.columns if c.startswith("y_")],
                        key=lambda s: int(s.split("_")[1]))
        Y = df[y_cols].to_numpy(dtype=np.float32)
        X = df[["x"]].to_numpy(dtype=np.float32)
        return X, Y, y_cols

    def make_seq(self, X, Y, T=32, stride=1):
        N = len(X)
        idx = list(range(0, N - T + 1, stride))
        X_seq = np.stack([X[i:i + T] for i in idx], axis=0)
        Y_seq = np.stack([Y[i:i + T] for i in idx], axis=0)
        return X_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, X_seq, Y_seq):
            self.X = torch.from_numpy(X_seq).float()
            self.Y = torch.from_numpy(Y_seq).float()

        def __len__(self):  return self.X.shape[0]

        def __getitem__(self, i):  return self.X[i], self.Y[i]

    def get_dataloader(self, df):
        df['ds'] = pd.to_datetime(df['ds'])

        # fix missing data and duplicate data and na
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
        df_wide['x'] = df_wide.filter(like="y_").sum(axis=1)

        X, Y, y_cols = self.extract_XY(df_wide)
        X_train, Y_train = X[:-1], Y[:-1]
        X_test, Y_test = X[-1:], Y[-1:]

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        X_std = sx.transform(X_train)
        Y_std = sy.transform(Y_train)

        X_seq, Y_seq = self.make_seq(X_std, Y_std, T=32, stride=1)
        ds = self.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        return (loader, sx, sy, X_train, Y_train, X_test, Y_test)

    def forecast(self, X_hist, X_future, T, sx=None, sy=None):
        self.eval()
        dev = next(self.parameters()).device
        with torch.no_grad():
            hist = X_hist[-T:]
            fut = np.asarray(X_future, dtype=np.float32)

            if sx is not None:
                hist = sx.transform(hist)
                fut = sx.transform(fut)

            x_in = np.concatenate([hist, fut], axis=0)[None, ...]
            x_in = torch.tensor(x_in, dtype=torch.float32, device=dev)

            o_all, _ = self(x_in)
            y_std = o_all[0, -len(X_future):].cpu().numpy()
            return sy.inverse_transform(y_std) if sy is not None else y_std


class RNN_FeatureAttention_wn(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4,
                 activation="tanh", learn_z0=True, dropout=0.0):
        super().__init__()
        self.h = 24

        self.W = nn.Parameter(torch.randn(self.h, self.h) * 0.01)
        self.U = nn.Parameter(torch.randn(self.h, input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.h))

        self.A = nn.Parameter(torch.randn(24, self.h) * 0.01)
        self.c = nn.Parameter(torch.zeros(24))

        self.feat_embed = nn.Parameter(torch.randn(24, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                          batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(self.h))

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

    def _feature_attend(self, z):
        B = z.shape[0]
        tokens = z.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.ln2(tokens + ffn_out)
        delta = self.out_proj(tokens).squeeze(-1)
        return z + delta

    def forward(self, x, z0=None):
        B, T, D = x.shape
        assert D == self.U.shape[1], f"Expected input_dim={self.U.shape[1]}, got {D}"

        if z0 is None:
            if self.learn_z0:
                z_t = self.z0.to(x.device).expand(B, self.h)
            else:
                z_t = torch.zeros(B, self.h, device=x.device)
        else:
            z_t = z0

        Z, O = [], []
        for t in range(T):
            x_t = x[:, t, :]
            z_t = self.act(z_t @ self.W.T + x_t @ self.U.T + self.b)
            z_t = self._feature_attend(z_t)
            Z.append(z_t)
            o_t = z_t @ self.A.T + self.c
            O.append(o_t)

        Z = torch.stack(Z, dim=1)
        O = torch.stack(O, dim=1)
        return O, Z

    def extract_XY(self, df):
        y_cols = sorted(
            [c for c in df.columns if c.startswith("y_")],
            key=lambda s: int(s.split("_")[1])
        )
        Y = df[y_cols].to_numpy(dtype=np.float32)
        X = df[["x_0", "x_1"]].to_numpy(dtype=np.float32)

        return X, Y, y_cols

    def make_seq(self, X, Y, T=32, stride=1):
        N = len(X)
        idx = list(range(0, N - T + 1, stride))
        X_seq = np.stack([X[i:i + T] for i in idx], axis=0)
        Y_seq = np.stack([Y[i:i + T] for i in idx], axis=0)
        return X_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, X_seq, Y_seq):
            self.X = torch.from_numpy(X_seq).float()
            self.Y = torch.from_numpy(Y_seq).float()

        def __len__(self):  return self.X.shape[0]

        def __getitem__(self, i):  return self.X[i], self.Y[i]

    def get_dataloader(self, df):
        df['ds'] = pd.to_datetime(df['ds'])

        # fix missing data and duplicate data and na
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
        df_wide['x_1'] = pd.to_datetime(df_wide['day']).dt.weekday

        X, Y, y_cols = self.extract_XY(df_wide)
        X_train, Y_train = X[:-1], Y[:-1]
        X_test, Y_test = X[-1:], Y[-1:]

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        X_std = sx.transform(X_train)
        Y_std = sy.transform(Y_train)

        X_seq, Y_seq = self.make_seq(X_std, Y_std, T=32, stride=1)
        ds = self.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        return (loader, sx, sy, X_train, Y_train, X_test, Y_test)

    def forecast(self, X_hist, X_future, T, sx=None, sy=None):
        self.eval()
        dev = next(self.parameters()).device
        with torch.no_grad():
            hist = X_hist[-T:]
            fut = np.asarray(X_future, dtype=np.float32)

            if sx is not None:
                hist = sx.transform(hist)
                fut = sx.transform(fut)

            x_in = np.concatenate([hist, fut], axis=0)[None, ...]
            x_in = torch.tensor(x_in, dtype=torch.float32, device=dev)

            o_all, _ = self(x_in)
            y_std = o_all[0, -len(X_future):].cpu().numpy()
        return sy.inverse_transform(y_std) if sy is not None else y_std


class RNN_FeatureAttention_wc(nn.Module):
    def __init__(self,
                 cont_dim=1,
                 cat_card=7,
                 cat_emb_dim=4,
                 d_model=128,
                 nhead=4,
                 activation="tanh",
                 learn_z0=True,
                 dropout=0.0,
                 H=24):
        super().__init__()
        self.h = H

        self.emb = nn.Embedding(num_embeddings=cat_card, embedding_dim=cat_emb_dim)

        input_dim = cont_dim + cat_emb_dim

        self.W = nn.Parameter(torch.randn(self.h, self.h) * 0.01)
        self.U = nn.Parameter(torch.randn(self.h, input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.h))

        self.A = nn.Parameter(torch.randn(24, self.h) * 0.01)
        self.c = nn.Parameter(torch.zeros(24))

        self.feat_embed = nn.Parameter(torch.randn(24, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                          batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(self.h))

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

    def _feature_attend(self, z):
        tokens = z.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.ln2(tokens + ffn_out)
        delta = self.out_proj(tokens).squeeze(-1)
        return z + delta

    def forward(self, x_cont, x_cat, z0=None):
        B, T, cont_dim = x_cont.shape
        assert x_cat.shape[:2] == (B, T), "x_cat must be (B,T)"

        if z0 is None:
            if self.learn_z0:
                z_t = self.z0.to(x_cont.device).expand(B, self.h)
            else:
                z_t = torch.zeros(B, self.h, device=x_cont.device)
        else:
            z_t = z0

        Z, O = [], []
        for t in range(T):
            x_cont_t = x_cont[:, t, :]
            emb_t = self.emb(x_cat[:, t])
            x_t = torch.cat([x_cont_t, emb_t], dim=-1)
            z_t = self.act(z_t @ self.W.T + x_t @ self.U.T + self.b)
            z_t = self._feature_attend(z_t)
            o_t = z_t @ self.A.T + self.c
            Z.append(z_t)
            O.append(o_t)

        Z = torch.stack(Z, dim=1)
        O = torch.stack(O, dim=1)
        return O, Z

    def extract_XY(self, df):
        y_cols = sorted(
            [c for c in df.columns if c.startswith("y_")],
            key=lambda s: int(s.split("_")[1])
        )
        Y = df[y_cols].to_numpy(dtype=np.float32)
        X = df[["x_0", "x_1"]].to_numpy(dtype=np.float32)

        return X, Y, y_cols

    def make_seq(self, Xc, Xk, Y, T=32, stride=1):
        N = len(Xc)
        idx = list(range(0, N - T + 1, stride))
        Xc_seq = np.stack([Xc[i:i + T] for i in idx], axis=0)
        Xk_seq = np.stack([Xk[i:i + T] for i in idx], axis=0)
        Y_seq = np.stack([Y[i:i + T] for i in idx], axis=0)
        return Xc_seq, Xk_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, Xc_seq, Xk_seq, Y_seq):
            self.Xc = torch.from_numpy(Xc_seq).float()
            self.Xk = torch.from_numpy(Xk_seq).long()
            self.Y = torch.from_numpy(Y_seq).float()

        def __len__(self):
            return self.Xc.shape[0]

        def __getitem__(self, i):
            return self.Xc[i], self.Xk[i], self.Y[i]

    def get_dataloader(self, df):
        df['ds'] = pd.to_datetime(df['ds'])

        # fix missing data and duplicate data and na
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
        df_wide['x_1'] = pd.to_datetime(df_wide['day']).dt.weekday

        X, Y, y_cols = self.extract_XY(df_wide)
        X_cont = X[:, [0]].astype(np.float32)
        X_cat = X[:, 1].astype(np.int64)
        Y = Y.astype(np.float32)
        Xc_train, Xc_test = X_cont[:-1], X_cont[-1:]
        Xk_train, Xk_test = X_cat[:-1], X_cat[-1:]
        Y_train, Y_test = Y[:-1], Y[-1:]

        sx = StandardScaler().fit(Xc_train)
        sy = StandardScaler().fit(Y_train)

        Xc_train_std = sx.transform(Xc_train).astype(np.float32)
        Y_train_std = sy.transform(Y_train).astype(np.float32)
        Xc_test_std = sx.transform(Xc_test).astype(np.float32)

        Xc_seq, Xk_seq, Y_seq = self.make_seq(Xc_train_std, Xk_train, Y_train_std, T=32, stride=1)
        ds = self.XYSeqDataset(Xc_seq, Xk_seq, Y_seq)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        return (loader, sx, sy, Xc_train, Xk_train, Y_train, Xc_test, Xk_test, Y_test)

    def forecast(self, Xc_hist, Xk_hist, Xc_fut, Xk_fut, T, sx=None, sy=None, cont_cols=(0,), cat_col=1):
        self.eval()
        dev = next(self.parameters()).device
        Xc_ctx = Xc_hist[-T:]
        Xk_ctx = Xk_hist[-T:]

        if sx is not None:
            Xc_ctx = sx.transform(Xc_ctx).astype(np.float32)
            Xc_fut = sx.transform(Xc_fut).astype(np.float32)

        Xc_in = np.concatenate([Xc_ctx, Xc_fut], axis=0)[None, ...]
        Xk_in = np.concatenate([Xk_ctx, Xk_fut], axis=0)[None, ...]

        x_cont = torch.from_numpy(Xc_in).to(dev)
        x_cat = torch.from_numpy(Xk_in).to(dev)

        with torch.no_grad():
            O_all, _ = self(x_cont, x_cat)
            H = len(Xc_fut)
            y_std = O_all[0, -H:, :].cpu().numpy()

        return sy.inverse_transform(y_std) if sy is not None else y_std


def append_fourier_features(df_wide: pd.DataFrame, K: int, period_days: float, mode: str = "vector"):
    if K <= 0:
        return df_wide, 0

    existing = [c for c in df_wide.columns if c.startswith("x_") and c != "x_0"]
    start_idx = 1 + len(existing)

    t0 = pd.to_datetime(df_wide['day']).astype('int64') // 86_400_000_000_000
    t_day = (t0 - t0.min()).astype(float).to_numpy()
    P = float(period_days)
    N, H = len(df_wide), 24
    added = 2 * K

    if mode == "vector":
        feats = np.empty((N, added), dtype=np.float64)
        for k in range(1, K + 1):
            w = 2 * np.pi * k / P
            feats[:, 2 * (k - 1)] = np.sin(w * t_day)
            feats[:, 2 * (k - 1) + 1] = np.cos(w * t_day)
        cols = [f"x_{start_idx + j}" for j in range(added)]
        feats_df = pd.DataFrame(feats, columns=cols, index=df_wide.index)
        return pd.concat([df_wide, feats_df], axis=1), added

    elif mode == "matrix":
        t_mat = t_day[:, None] + np.arange(H)[None, :] / 24.0
        S = np.stack([np.sin(2 * np.pi * k / P * t_mat) for k in range(1, K + 1)], axis=0)
        C = np.stack([np.cos(2 * np.pi * k / P * t_mat) for k in range(1, K + 1)], axis=0)

        blocks = []
        for h in range(H):
            blk = np.empty((N, added), dtype=np.float64)
            blk[:, 0::2] = S[:, :, h].transpose(1, 0)
            blk[:, 1::2] = C[:, :, h].transpose(1, 0)
            blocks.append(blk)
        Xf = np.concatenate(blocks, axis=1)

        cols = [f"x_{start_idx + j}" for j in range(Xf.shape[1])]
        feats_df = pd.DataFrame(Xf, columns=cols, index=df_wide.index)
        return pd.concat([df_wide, feats_df], axis=1), added

    else:
        raise ValueError("mode must be 'vector' or 'matrix'")


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

    def __init__(
            self,
            cont_dim=0,
            fourier_dim=0,
            xf_mode="vector",
            d_model=128,
            nhead=4,
            activation="relu",
            learn_z0=True,
            dropout=0.0,
            H=24,
            use_gate=True,
            nonneg_U0=False,
    ):
        super().__init__()
        assert xf_mode in ("vector", "matrix")
        if xf_mode == "vector":
            assert cont_dim == 1 + fourier_dim, "vector mode cont_dim should be 1+F"
        else:
            assert cont_dim == 1 + 24 * fourier_dim, "matrix mode cont_dim should be 1+24*F"

        self.h = H
        self.F = fourier_dim
        self.mode = xf_mode
        self.nonneg_U0 = nonneg_U0
        self.cont_dim = cont_dim

        self.W = nn.Parameter(torch.randn(self.h, self.h) * 0.01)
        self.U0_raw = nn.Parameter(torch.randn(self.h, 1) * 0.01)
        self.Uf = nn.Parameter(torch.randn(self.h, self.F) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.h))

        self.A = nn.Parameter(torch.randn(self.h, self.h) * 0.01)
        self.c = nn.Parameter(torch.zeros(self.h))

        self.feat_embed = nn.Parameter(torch.randn(self.h, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(nn.Linear(self.h, self.h), nn.Sigmoid())

        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(self.h))

        self.act = F.relu if activation == "relu" else torch.tanh

    def _feature_attend(self, z):
        tokens = z.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.ffn(tokens))
        delta = self.out_proj(tokens).squeeze(-1)
        return z + (self.gate(z) * delta if self.use_gate else delta)

    def reg_loss(self, lambda0=0.0, lambdaf=0.0, harmonic_orders=None):
        loss = torch.tensor(0.0, device=self.W.device)
        U0 = F.softplus(self.U0_raw) if self.nonneg_U0 else self.U0_raw
        if lambda0 > 0:
            loss = loss + lambda0 * (U0 ** 2).sum()
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
        if self.mode == "vector":
            xf_vec = x_cont[:, :, 1:]
        else:
            xf_flat = x_cont[:, :, 1:]
            xf_mat = xf_flat.view(B, T, 24, self.F)

        if z0 is None:
            z_t = self.z0.expand(B, self.h) if hasattr(self, "z0") else torch.zeros(B, self.h, device=x_cont.device)
        else:
            z_t = z0

        Z, O = [], []
        for t in range(T):
            x0_t = x0[:, t, :]
            U0 = F.softplus(self.U0_raw) if self.nonneg_U0 else self.U0_raw
            pre = z_t @ self.W.T + x0_t @ U0.T + self.b

            if self.mode == "vector":
                xf_t = xf_vec[:, t, :]
                pre = pre + xf_t @ self.Uf.T
            else:
                Xf_t = xf_mat[:, t, :, :]
                pre = pre + (Xf_t * self.Uf.unsqueeze(0)).sum(-1)

            z_t = self.act(pre)
            z_t = self._feature_attend(z_t)
            Z.append(z_t)

            o_t = z_t @ self.A.T + self.c
            O.append(o_t)

        Z = torch.stack(Z, dim=1)
        O = torch.stack(O, dim=1)
        return O, Z

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

        def __len__(self): return self.X.shape[0]

        def __getitem__(self, i): return self.X[i], self.Y[i]

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

    def get_dataloader(self, df, fourier_config):
        df['ds'] = pd.to_datetime(df['ds'])

        # fix missing data and duplicate data and na
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



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RNN_fourier_GRU(nn.Module):
    """
    Minimal-change RNN_fourier with an optional GRU backbone.

    - Keeps your self-attention feature layer, gating, and output head (A, c).
    - If use_gru=True: uses a single nn.GRU over the full sequence x_cont [B,T,C].
      No x-splitting; self-attention and output are vectorized over time.
    - If use_gru=False: preserves the original hand-written recurrence:
        z_t = act(z_t @ W^T + x0_t @ U0^T + (xf-part) + b)
        z_t = self._feature_attend(z_t)
        o_t = z_t @ A^T + c

    Shapes:
      x_cont: [B, T, cont_dim]
      returns:
        O: [B, T, H]    (mapped via A, c; H is typically 24 for hourly)
        Z: [B, T, H]    (the attended latent states)
    """

    def __init__(
        self,
        cont_dim: int = 0,             # total input feature dimension per time step
        fourier_dim: int = 0,          # F: number of Fourier/exogenous dims used by the hand-rolled branch
        xf_mode: str = "vector",       # "vector" or "matrix" (matrix expects reshape to [24, F] per t)
        d_model: int = 128,            # attention model dimension
        nhead: int = 4,                # attention heads
        activation: str = "relu",      # "relu" or "tanh" for the hand-rolled path
        learn_z0: bool = True,
        dropout: float = 0.0,
        H: int = 24,                   # latent (and output) dimension
        use_gate: bool = True,
        nonneg_U0: bool = False,
        use_gru: bool = False,         # <— turn on GRU backbone (full x, no splitting)
    ):
        super().__init__()
        self.h = H
        self.F = fourier_dim
        self.mode = xf_mode
        self.nonneg_U0 = nonneg_U0
        self.cont_dim = cont_dim
        self.use_gru = use_gru

        # ===== Hand-written recurrence params (kept for compatibility) =====
        # Note: these are not used when use_gru=True, but we keep them so old checkpoints still load.
        self.W = nn.Parameter(torch.randn(self.h, self.h) * 0.01)          # [H, H]
        self.U0_raw = nn.Parameter(torch.randn(self.h, 1) * 0.01)          # [H, 1]
        self.Uf = nn.Parameter(torch.randn(self.h, self.F) * 0.01)         # [H, F]
        self.b = nn.Parameter(torch.zeros(self.h))                          # [H]

        # ===== Output head (kept) =====
        self.A = nn.Parameter(torch.randn(self.h, self.h) * 0.01)          # [H, H]
        self.c = nn.Parameter(torch.zeros(self.h))                          # [H]

        # ===== Self-attention feature layer (kept) =====
        # Treat each latent dimension as a token; feat_embed maps H tokens -> d_model features
        self.feat_embed = nn.Parameter(torch.randn(self.h, d_model) * 0.02)  # [H, d_model]
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)   # projects per-token back to a scalar delta
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(nn.Linear(self.h, self.h), nn.Sigmoid())

        # ===== Initial hidden state (kept) =====
        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(self.h))  # [H]

        # ===== Activation for hand-rolled path =====
        self.act = F.relu if activation == "relu" else torch.tanh

        # ===== GRU backbone (sequence-wise) =====
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=self.cont_dim,    # full x features per step
                hidden_size=self.h,          # latent size == H
                num_layers=1,
                batch_first=True,            # x_cont is [B, T, C]
                bidirectional=False,
                dropout=0.0,
            )

    # ---------- Self-attention over latent tokens (per time step) ----------
    def _feature_attend(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, H] latent vector; each of the H dims is treated as a 'token'.
        Returns: [B, H] updated z with residual from MHA+FFN (and optional gate).
        """
        # Map tokens: z[:, i] * feat_embed[i] -> token i's d_model features
        # tokens: [B, H, d_model]
        tokens = z.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)     # [B, H, d_model]
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.ffn(tokens))
        delta = self.out_proj(tokens).squeeze(-1)           # [B, H]
        if self.use_gate:
            return z + self.gate(z) * delta
        return z + delta

    def _feature_attend_seq(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Vectorized attention across all time steps.
        z_seq: [B, T, H] -> returns [B, T, H]
        """
        B, T, H = z_seq.shape
        z_flat = z_seq.reshape(B * T, H)        # [B*T, H]
        z_flat = self._feature_attend(z_flat)   # [B*T, H]
        return z_flat.view(B, T, H)             # [B, T, H]

    # ---------- Optional regularization carried over from your original code ----------
    def reg_loss(self, lambda0: float = 0.0, lambdaf: float = 0.0, harmonic_orders: Optional[torch.Tensor] = None):
        """
        L2-style penalty on U0 and Uf (kept for compatibility with existing training scripts).
        When use_gru=True, you can simply set both lambdas to 0 to disable.
        """
        device = self.W.device
        loss = torch.tensor(0.0, device=device)
        U0 = F.softplus(self.U0_raw) if self.nonneg_U0 else self.U0_raw  # ensure positivity if requested

        if lambda0 > 0:
            loss = loss + lambda0 * (U0 ** 2).sum()

        if lambdaf > 0:
            if harmonic_orders is None:
                loss = loss + lambdaf * (self.Uf ** 2).sum()
            else:
                # weight by squared harmonic orders if provided
                w = torch.as_tensor(harmonic_orders, dtype=self.Uf.dtype, device=device)  # [F]
                loss = loss + lambdaf * ((self.Uf ** 2) * (w ** 2).unsqueeze(0)).sum()

        return loss

    # ---------- Forward ----------
    def forward(self, x_cont: torch.Tensor, z0: Optional[torch.Tensor] = None):
        """
        x_cont: [B, T, cont_dim] full feature sequence.
        z0: optional initial latent state [B, H] (used to seed the GRU hidden state or the hand-rolled z_t).
        returns:
          O: [B, T, H]
          Z: [B, T, H]
        """
        B, T, C = x_cont.shape
        assert C == self.cont_dim, f"Expected cont_dim={self.cont_dim}, got {C}"

        # ----- GRU path: no splitting, process the entire sequence in one call -----
        if self.use_gru:
            # Prepare initial hidden state for GRU if provided
            h0 = None
            if z0 is not None:
                h0 = z0.unsqueeze(0)  # [1, B, H]

            # 1) GRU over the full sequence
            h_seq, _ = self.gru(x_cont, h0)     # [B, T, H]

            # 2) Self-attention per step (vectorized)
            z_seq = self._feature_attend_seq(h_seq)  # [B, T, H]

            # 3) Output head per step
            O = z_seq @ self.A.T + self.c          # [B, T, H]
            Z = z_seq
            return O, Z

        # ----- Hand-rolled recurrence path (original logic kept) -----
        # We split x only for this branch, to remain faithful to the original design.
        x0 = x_cont[:, :, :1]  # [B, T, 1]
        if self.mode == "vector":
            xf_vec = x_cont[:, :, 1:]  # [B, T, F]
        else:
            # Expect the remainder to be [B, T, 24*F] that reshapes to [B, T, 24, F]
            xf_flat = x_cont[:, :, 1:]             # [B, T, 24*F]
            assert xf_flat.size(-1) == 24 * self.F, (
                f"xf_mode='matrix' expects last dim 24*F, got {xf_flat.size(-1)} vs 24*{self.F}"
            )
            xf_mat = xf_flat.view(B, T, 24, self.F)

        # Initialize z_t
        if z0 is not None:
            z_t = z0  # [B, H]
        else:
            z_t = self.z0.expand(B, self.h) if hasattr(self, "z0") else torch.zeros(B, self.h, device=x_cont.device)

        Z_list, O_list = [], []
        for t in range(T):
            x0_t = x0[:, t, :]                                # [B, 1]
            U0 = F.softplus(self.U0_raw) if self.nonneg_U0 else self.U0_raw  # [H, 1]

            pre = z_t @ self.W.T + x0_t @ U0.T + self.b       # [B, H]

            if self.mode == "vector":
                if self.F > 0:
                    xf_t = xf_vec[:, t, :]                    # [B, F]
                    pre = pre + xf_t @ self.Uf.T              # [B, H]
            else:
                # xf_mat: [B, T, 24, F], Uf: [H, F]
                # broadcast sum over F to produce [B, 24], then add to pre (assumes 24==H)
                # (Xf_t * Uf.unsqueeze(0)).sum(-1) -> [B, 24]
                Xf_t = xf_mat[:, t, :, :]                     # [B, 24, F]
                add_ = (Xf_t * self.Uf.unsqueeze(0)).sum(-1)  # [B, 24]
                # If H != 24 this addition is ill-defined; keep the original assumption H==24.
                if add_.shape[-1] != pre.shape[-1]:
                    raise ValueError(f"xf_mode='matrix' requires H==24; got add_.shape[-1]={add_.shape[-1]} vs H={self.h}")
                pre = pre + add_

            # nonlinearity
            z_t = self.act(pre)                               # [B, H]

            # self-attention feature layer
            z_t = self._feature_attend(z_t)                   # [B, H]
            Z_list.append(z_t)

            # output head
            o_t = z_t @ self.A.T + self.c                     # [B, H]
            O_list.append(o_t)

        Z = torch.stack(Z_list, dim=1)  # [B, T, H]
        O = torch.stack(O_list, dim=1)  # [B, T, H]
        return O, Z

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

        def __len__(self): return self.X.shape[0]

        def __getitem__(self, i): return self.X[i], self.Y[i]

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

    def get_dataloader(self, df, fourier_config):
        df['ds'] = pd.to_datetime(df['ds'])

        # fix missing data and duplicate data and na
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


from asym_helper import AsymmetricFeatureAttention


class RNN_asym_feature(nn.Module):
    def __init__(self, d_model=128, nhead=4, activation="tanh",
                 learn_z0=True, dropout=0.0, H=24):
        super().__init__()
        self.h = H
        self.W = nn.Parameter(torch.randn(H, H) * 0.01)
        self.U = nn.Parameter(torch.randn(H, 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(H))
        self.activation = torch.tanh if activation == "tanh" else F.relu
        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(H))

        self.feat_embed = nn.Parameter(torch.randn(H, d_model) * 0.01)

        self.asym_attn = AsymmetricFeatureAttention(
            H=H, d_model=d_model, nhead=nhead, dropout=dropout,
            use_dir_bias=True, separate_dir_bias=True, use_softmax_gating=True
        )

        self.A = nn.Parameter(torch.randn(H, H) * 0.01)
        self.c = nn.Parameter(torch.zeros(H))

    def forward(self, X):
        B, T, _ = X.shape
        z_prev = self.z0.expand(B, -1) if self.learn_z0 else torch.zeros(B, self.h, device=X.device)
        O, Z = [], []
        for t in range(T):
            x_t = X[:, t, :]
            z = self.activation(z_prev @ self.W.T + x_t @ self.U.T + self.b)
            delta, aux = self.asym_attn(z, self.feat_embed)
            z = z + delta
            o = z @ self.A.T + self.c
            O.append(o);
            Z.append(z);
            z_prev = z
        return torch.stack(O, 1), torch.stack(Z, 1)

    def extract_XY(self, df):
        y_cols = sorted([c for c in df.columns if c.startswith("y_")],
                        key=lambda s: int(s.split("_")[1]))
        Y = df[y_cols].to_numpy(dtype=np.float32)
        X = df[["x"]].to_numpy(dtype=np.float32)
        return X, Y, y_cols

    def make_seq(self, X, Y, T=32, stride=1):
        N = len(X)
        idx = list(range(0, N - T + 1, stride))
        X_seq = np.stack([X[i:i + T] for i in idx], axis=0)
        Y_seq = np.stack([Y[i:i + T] for i in idx], axis=0)
        return X_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, X_seq, Y_seq):
            self.X = torch.from_numpy(X_seq).float()
            self.Y = torch.from_numpy(Y_seq).float()

        def __len__(self):  return self.X.shape[0]

        def __getitem__(self, i):  return self.X[i], self.Y[i]

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
        df_wide['x'] = df_wide.filter(like="y_").sum(axis=1)

        X, Y, y_cols = self.extract_XY(df_wide)
        X_train, Y_train = X[:-1], Y[:-1]
        X_test, Y_test = X[-1:], Y[-1:]

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        X_std = sx.transform(X_train)
        Y_std = sy.transform(Y_train)

        X_seq, Y_seq = self.make_seq(X_std, Y_std, T=32, stride=1)
        ds = self.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        return (loader, sx, sy, X_train, Y_train, X_test, Y_test)

    def forecast(self, X_hist, X_future, T, sx=None, sy=None):
        self.eval()
        dev = next(self.parameters()).device
        with torch.no_grad():
            hist = X_hist[-T:]
            fut = np.asarray(X_future, dtype=np.float32)

            if sx is not None:
                hist = sx.transform(hist)
                fut = sx.transform(fut)

            x_in = np.concatenate([hist, fut], axis=0)[None, ...]
            x_in = torch.tensor(x_in, dtype=torch.float32, device=dev)

            o_all, _ = self(x_in)
            y_std = o_all[0, -len(X_future):].cpu().numpy()
            return sy.inverse_transform(y_std) if sy is not None else y_std


from arma_helper import ARMAFeatureBlock, ARMAFeatureBlockSafeMPS


class RNN_ARMA_Feature(nn.Module):
    def __init__(self, H=24, d_model=128, activation="tanh", learn_z0=True):
        super().__init__()
        self.h = H
        self.W = nn.Parameter(torch.randn(H, H) * 0.01)
        self.U = nn.Parameter(torch.randn(H, 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(H))
        self.act = torch.tanh if activation == "tanh" else F.relu
        self.learn_z0 = learn_z0
        if learn_z0:
            self.z0 = nn.Parameter(torch.zeros(H))

        self.arma = ARMAFeatureBlock(H=H, K_ar=5, K_ma=5, gate="softmax", init_zero_delta=True, norm_type="layer")
        # self.arma = ARMAFeatureBlockSafeMPS(H=24, K_ar=5, K_ma=5, gate="softmax")

        self.A = nn.Parameter(torch.randn(H, H) * 0.01)
        self.c = nn.Parameter(torch.zeros(H))

    def forward(self, X):
        B, T, _ = X.shape
        z_prev = self.z0.expand(B, -1) if self.learn_z0 else torch.zeros(B, self.h, device=X.device)
        O, Z = [], []
        for t in range(T):
            x_t = X[:, t, :]
            z_t = self.act(z_prev @ self.W.T + x_t @ self.U.T + self.b)
            z_t_new, delta, aux = self.arma(z_prev, z_t)
            o_t = z_t_new @ self.A.T + self.c

            O.append(o_t);
            Z.append(z_t_new)
            z_prev = z_t_new

        return torch.stack(O, 1), torch.stack(Z, 1)

    def extract_XY(self, df):
        y_cols = sorted([c for c in df.columns if c.startswith("y_")],
                        key=lambda s: int(s.split("_")[1]))
        Y = df[y_cols].to_numpy(dtype=np.float32)
        X = df[["x"]].to_numpy(dtype=np.float32)
        return X, Y, y_cols

    def make_seq(self, X, Y, T=32, stride=1):
        N = len(X)
        idx = list(range(0, N - T + 1, stride))
        X_seq = np.stack([X[i:i + T] for i in idx], axis=0)
        Y_seq = np.stack([Y[i:i + T] for i in idx], axis=0)
        return X_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, X_seq, Y_seq):
            self.X = torch.from_numpy(X_seq).float()
            self.Y = torch.from_numpy(Y_seq).float()

        def __len__(self):  return self.X.shape[0]

        def __getitem__(self, i):  return self.X[i], self.Y[i]

    def get_dataloader(self, df):
        df['ds'] = pd.to_datetime(df['ds'])

        # fix missing data and duplicate data and na
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
        df_wide['x'] = df_wide.filter(like="y_").sum(axis=1)

        X, Y, y_cols = self.extract_XY(df_wide)
        X_train, Y_train = X[:-1], Y[:-1]
        X_test, Y_test = X[-1:], Y[-1:]

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        X_std = sx.transform(X_train)
        Y_std = sy.transform(Y_train)

        X_seq, Y_seq = self.make_seq(X_std, Y_std, T=32, stride=1)
        ds = self.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        return (loader, sx, sy, X_train, Y_train, X_test, Y_test)

    def forecast(self, X_hist, X_future, T, sx=None, sy=None):
        self.eval()
        dev = next(self.parameters()).device
        with torch.no_grad():
            hist = X_hist[-T:]
            fut = np.asarray(X_future, dtype=np.float32)

            if sx is not None:
                hist = sx.transform(hist)
                fut = sx.transform(fut)

            x_in = np.concatenate([hist, fut], axis=0)[None, ...]
            x_in = torch.tensor(x_in, dtype=torch.float32, device=dev)

            o_all, _ = self(x_in)
            y_std = o_all[0, -len(X_future):].cpu().numpy()
            return sy.inverse_transform(y_std) if sy is not None else y_std


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


class RNN_train_1:
    def __init__(self, model, training_config):
        self.model = model
        self.training_config = training_config
        self.model.to(self.training_config.device)

    def __call__(self, df):
        loader, sx, sy, X_train, Y_train, X_test, Y_test = self.model.get_dataloader(df)
        opt = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        for epoch in range(self.training_config.n_epochs):
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(self.training_config.device)
                yb = yb.to(self.training_config.device)
                opt.zero_grad()
                o, _ = self.model(xb)
                loss = F.mse_loss(o, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            print(f"epoch {epoch + 1} loss:", float(loss))
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.forecast(X_train, X_test, T=32, sx=sx, sy=sy)
        y_hat_flat = y_pred.flatten()
        y_flat = Y_test.flatten()
        return y_hat_flat, y_flat


class RNN_train_2:
    def __init__(self, model, training_config):
        self.model = model
        self.training_config = training_config
        self.model.to(self.training_config.device)

    def __call__(self, df):
        loader, sx, sy, Xc_train, Xk_train, Y_train, Xc_test, Xk_test, Y_test = self.model.get_dataloader(df)
        opt = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        for epoch in range(self.training_config.n_epochs):
            self.model.train()
            for Xc_b, Xk_b, Y_b in loader:
                Xc_b = Xc_b.to(self.training_config.device)
                Xk_b = Xk_b.to(self.training_config.device)
                Y_b = Y_b.to(self.training_config.device)
                opt.zero_grad()
                O, _ = self.model(Xc_b, Xk_b)
                loss = F.mse_loss(O, Y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            print(f"epoch {epoch + 1} loss:", float(loss))
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.forecast(Xc_train, Xk_train, Xc_test, Xk_test, T=32, sx=sx, sy=sy)
        y_hat_flat = y_pred.flatten()
        y_flat = Y_test.flatten()
        return y_hat_flat, y_flat


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
        opt = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        for ep in range(self.training_config.n_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                o, _ = self.model(xb)
                data_loss = F.mse_loss(o, yb)
                prior_loss = self.model.reg_loss(lambda0=self.training_config.lambda0,
                                                 lambdaf=self.training_config.lambdaf,
                                                 harmonic_orders=harmonic_orders)
                loss = data_loss + prior_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
            print(f"epoch {ep + 1} loss: {float(loss):.4f}")

        y_pred = self.model.forecast_knownX(X_hist=X_train, X_future=X_test, T=32, sx=sx, sy=sy)
        return list(y_pred.flatten()), list(Y_test.flatten())
