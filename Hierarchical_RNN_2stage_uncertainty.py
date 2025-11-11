import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Optional, Literal, Tuple, List, Dict
from dataclasses import dataclass
import calendar
from datetime import datetime


# ============================================================================
# Utility Functions
# ============================================================================

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


def get_base_model(model):
    """Get the base model from DataParallel wrapper if needed."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def simple_dst_fix(df: pd.DataFrame, start_at_midnight: bool = True) -> pd.DataFrame:
    """Fix DST issues and missing data in hourly time series."""
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


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return calendar.isleap(year)


def get_days_in_year(year: int) -> int:
    """Get number of days in a year (365 or 366)."""
    return 366 if is_leap_year(year) else 365


# ============================================================================
# Fourier Feature Generation
# ============================================================================

def append_fourier_features_matrix(df_wide, K, period_days, H, mode="matrix", start_idx=None):
    if K <= 0:
        return df_wide, 0

    existing = [c for c in df_wide.columns if c.startswith("x_") and c != "x_0"]
    if start_idx is None:
        start_idx = 1 + len(existing)

    # 使用固定的参考日期（例如 2000-01-01）
    reference_date = pd.Timestamp('2000-01-01')
    t0 = pd.to_datetime(df_wide["ds"]).astype("int64") // 86_400_000_000_000
    t_ref = (reference_date.value // 10 ** 9) // 86400  # 转换为天数
    t_day = (t0 - t_ref).astype(float).to_numpy()  # 使用固定参考点

    P = float(period_days)
    N = len(df_wide)

    if mode == "matrix":
        # Special handling for yearly to daily (H=366)
        if H == 366:
            # For yearly data, compute features for days 0-365 of the year
            # not days offset from the input date
            years = pd.to_datetime(df_wide["ds"]).dt.year.to_numpy()
            t_mat = np.zeros((N, H))
            for i in range(N):
                year_start = pd.Timestamp(f'{years[i]}-01-01')
                year_start_days = (year_start.value // 10 ** 9) // 86400
                # Days 0-365 of this specific year
                t_mat[i, :] = year_start_days - t_ref + np.arange(H)
        else:
            # For other cases (e.g., daily to hourly), use the original logic
            t_mat = t_day[:, None] + np.arange(H)[None, :]

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
        raise ValueError("Only 'matrix' mode is supported")


# ============================================================================
# Year Length Selection Head
# ============================================================================

class YearLengthHead(nn.Module):
    """
    Selection head that maps fixed 366-day output to variable year lengths.
    Uses separate linear layers for 365 and 366 day years.
    """

    def __init__(self, latent_dim=24):
        super().__init__()
        self.latent_dim = latent_dim

        self.head_365 = nn.Linear(latent_dim * 366, 365)  # Regular year
        self.head_366 = nn.Linear(latent_dim * 366, 366)  # Leap year

    def forward(self, z_366, year):
        """
        Args:
            z_366: (B, 366, latent_dim) - fixed output from RNN
            year: int or array - year

        Returns:
            Variable length daily output based on year
        """
        B = z_366.shape[0]
        z_flat = z_366.reshape(B, -1)  # (B, 366 * latent_dim)

        if is_leap_year(year):
            return self.head_366(z_flat)
        else:
            return self.head_365(z_flat)


# ============================================================================
# Base RNN Model
# ============================================================================

class RNN_downscaler(nn.Module):
    """
    Generalized RNN downscaler for any resolution level.
    """

    def __init__(
            self,
            cont_dim=0,
            fourier_dim=0,
            H=24,
            latent_dim=24,
            d_model=128,
            nhead=4,
            activation="relu",
            learn_z0=True,
            dropout=0.0,
            use_gate=True,
            rnn_type="rnn"
    ):
        super().__init__()

        assert cont_dim == 1 + H * fourier_dim, f"cont_dim should be 1 + {H} * {fourier_dim} = {1 + H * fourier_dim}"

        self.H = H
        self.latent_dim = latent_dim
        self.F = fourier_dim
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
        """Apply feature-wise attention mechanism."""
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
        """Regularization loss on Fourier coefficients."""
        loss = torch.tensor(0.0, device=self.Uf.device)

        if lambdaf > 0:
            if harmonic_orders is None:
                loss = loss + lambdaf * (self.Uf ** 2).sum()
            else:
                w = torch.as_tensor(harmonic_orders, dtype=self.Uf.dtype, device=self.Uf.device)
                loss = loss + lambdaf * ((self.Uf ** 2) * (w ** 2).unsqueeze(0)).sum()

        return loss

    def forward(self, x_cont, z0=None, return_hidden=False):
        """
        Forward pass with proper sequence handling.

        Args:
            x_cont: (B, T, C) input tensor
            z0: Optional initial hidden state
            return_hidden: If True, return the final hidden state for continuity
        """
        B, T, C = x_cont.shape
        assert C == self.cont_dim, f"expect cont_dim={self.cont_dim}, got {C}"

        x0 = x_cont[:, :, :1]
        xf = x_cont[:, :, 1:]

        # Initialize hidden state
        if z0 is None:
            if self.learn_z0:
                # For sequences, expand z0 properly
                h0 = self.z0.expand(-1, B, -1).contiguous()
            else:
                # Zero initialization if not learning z0
                h0 = torch.zeros(1, B, self.latent_dim, device=x_cont.device)
        else:
            h0 = z0

        # Process through RNN - this builds up hidden states over the sequence
        h, h_final = self.rnn(x0, h0)

        # Fourier contribution
        xf_mat = xf.view(B, T, self.H, self.F)
        fourier_h = (xf_mat * self.Uf.unsqueeze(0).unsqueeze(0)).sum(-1)

        if self.fourier_proj is not None:
            fourier_contrib = self.fourier_proj(fourier_h)
        else:
            fourier_contrib = fourier_h

        z = h + fourier_contrib

        z = self._feature_attend(z)

        o = z @ self.A.T + self.c

        if return_hidden:
            return o, z, h_final
        else:
            return o, z


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class FourierConfig:
    """Configuration for Fourier features at each level."""
    # Yearly → Daily
    K_yearly_to_daily: int = 8
    K_monthly_to_daily: int = 6
    K_weekly_to_daily: int = 6

    # Daily → Hourly
    K_daily_to_hourly: int = 8
    K_weekly_to_hourly: int = 3
    K_yearly_to_hourly: int = 2
    K_monthly_to_hourly: int = 2

    # Periods (in days)
    P_year: float = 365.25
    P_month: float = 365.25 / 12.0
    P_week: float = 7.0
    P_day: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    base_epochs: int = 50
    yearly_to_daily_multiplier: int = 12
    daily_to_hourly_multiplier: int = 1

    lr: float = 5e-4
    batch_size: int = 64

    lambda0: float = 5e-6
    lambdaf: float = 1e-3

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T_seq_yearly: int = 2  # For yearly->daily (limited data)
    T_seq_daily: int = 32  # For daily->hourly (plenty of data)
    stride: int = 1

    curriculum_start_prob: float = 0.5
    curriculum_end_prob: float = 0.25


# ============================================================================
# 2-Stage Hierarchical Downscaler
# ============================================================================

class HierarchicalDownscaler_2Stage(nn.Module):
    """
    2-stage hierarchical downscaling system: Yearly → Daily → Hourly
    """

    def __init__(
            self,
            fourier_config: FourierConfig,
            training_config: TrainingConfig,
            latent_dim: int = 24,
            d_model: int = 128,
            nhead: int = 4
    ):
        super().__init__()
        self.fc = fourier_config
        self.tc = training_config
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead

        self._init_models()

        self.scalers = {
            'yearly': {'sx': None, 'sy': None},
            'daily': {'sx': None, 'sy': None}
        }

    def _init_models(self):
        """Initialize the two RNN branches."""

        # Branch 1: Yearly → Daily (366 fixed outputs + selection head)
        fourier_dim_yearly = (
                self.fc.K_yearly_to_daily * 2 +
                self.fc.K_monthly_to_daily * 2 +
                self.fc.K_weekly_to_daily * 2
        )
        self.rnn_daily_base = RNN_downscaler(
            cont_dim=1 + 366 * fourier_dim_yearly,
            fourier_dim=fourier_dim_yearly,
            H=366,
            latent_dim=10*self.latent_dim,
            d_model=self.d_model,
            nhead=self.nhead
        )

        # Year length selection head
        self.year_head = YearLengthHead(latent_dim=self.latent_dim)

        # Branch 2: Daily → Hourly (24 outputs)
        fourier_dim_daily = (
                self.fc.K_yearly_to_hourly * 2 +
                self.fc.K_monthly_to_hourly * 2 +
                self.fc.K_daily_to_hourly * 2 +
                self.fc.K_weekly_to_hourly * 2
        )
        self.rnn_hourly = RNN_downscaler(
            cont_dim=1 + 24 * fourier_dim_daily,
            fourier_dim=fourier_dim_daily,
            H=24,
            latent_dim=self.latent_dim,
            d_model=self.d_model,
            nhead=self.nhead
        )

    def prepare_data(self, df_hourly: pd.DataFrame):
        """
        Prepare hierarchical training data from hourly ground truth.
        """
        df_hourly = simple_dst_fix(df_hourly)
        df_hourly['ds'] = pd.to_datetime(df_hourly['ds'])

        # Create yearly data
        df_hourly['year'] = df_hourly['ds'].dt.year
        df_yearly = df_hourly.groupby('year').agg({'y': 'sum', 'ds': 'first'})
        df_yearly = df_yearly.rename(columns={'y': 'yearly_sum'})
        df_yearly = df_yearly.sort_values('year').reset_index(drop=False)

        # Create daily data
        df_hourly['date'] = df_hourly['ds'].dt.date
        df_daily = df_hourly.groupby('date').agg({
            'y': 'sum',
            'ds': 'first'
        }).reset_index()
        df_daily['year'] = df_daily['ds'].dt.year
        df_daily = df_daily.rename(columns={'y': 'daily_sum'})
        df_daily = df_daily.sort_values('date').reset_index(drop=True)

        return {
            'hourly': df_hourly,
            'daily': df_daily,
            'yearly': df_yearly
        }

    def prepare_yearly_to_daily_data(self, df_yearly: pd.DataFrame, df_daily: pd.DataFrame):
        """Prepare training data for Branch 1: Yearly → Daily."""

        data = []

        for _, year_row in df_yearly.iterrows():
            year = year_row['year']
            yearly_sum = year_row['yearly_sum']

            # Get daily values for this year
            year_mask = df_daily['year'] == year
            daily_values = df_daily[year_mask]['daily_sum'].values

            days_in_year = get_days_in_year(year)

            if len(daily_values) == days_in_year:
                row_dict = {
                    'ds': year_row['ds'],
                    'year': year,
                    'x_0': yearly_sum,
                    'days_in_year': days_in_year
                }

                # Store all daily values
                for i in range(days_in_year):
                    row_dict[f'y_{i}'] = daily_values[i]

                # Pad with zeros to 366 days
                for i in range(days_in_year, 366):
                    row_dict[f'y_{i}'] = 0.0

                data.append(row_dict)

        df_wide = pd.DataFrame(data).sort_values('ds')

        # Add Fourier features
        start_idx = 1

        if self.fc.K_yearly_to_daily > 0:
            df_wide, n_feats = append_fourier_features_matrix(
                df_wide,
                K=self.fc.K_yearly_to_daily,
                period_days=self.fc.P_year,
                H=366,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        if self.fc.K_monthly_to_daily > 0:
            df_wide, n_feats = append_fourier_features_matrix(
                df_wide,
                K=self.fc.K_monthly_to_daily,
                period_days=365.25 / 12,  # Monthly period
                H=366,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        if self.fc.K_weekly_to_daily > 0:
            df_wide, n_feats = append_fourier_features_matrix(
                df_wide,
                K=self.fc.K_weekly_to_daily,
                period_days=self.fc.P_week,
                H=366,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        return df_wide.sort_values('ds').reset_index(drop=True)

    def prepare_daily_to_hourly_data(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame):
        """Prepare training data for Branch 2: Daily → Hourly."""

        data = []

        for _, day_row in df_daily.iterrows():

            if 'date' in day_row:
                date = day_row['date']
            else:
                date = pd.to_datetime(day_row['ds']).date()

            daily_sum = day_row['daily_sum']

            day_mask = df_hourly['ds'].dt.date == date
            hourly_values = df_hourly[day_mask]['y'].values

            if len(hourly_values) == 24:
                row_dict = {
                    'ds': day_row['ds'],
                    'date': date,
                    'x_0': daily_sum
                }

                for i in range(24):
                    row_dict[f'y_{i}'] = hourly_values[i]

                data.append(row_dict)

        df_wide = pd.DataFrame(data)

        # Add Fourier features
        start_idx = 1

        # 1. Yearly
        if self.fc.K_yearly_to_hourly > 0:
            df_wide, n_feats = append_fourier_features_matrix(
                df_wide,
                K=self.fc.K_yearly_to_hourly,
                period_days=self.fc.P_year,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        # 2. Monthly
        if self.fc.K_monthly_to_hourly > 0:
            df_wide, n_feats = append_fourier_features_matrix(
                df_wide,
                K=self.fc.K_monthly_to_hourly,
                period_days=self.fc.P_month,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        # 3. Weekly
        if self.fc.K_weekly_to_hourly > 0:
            df_wide, n_feats = append_fourier_features_matrix(
                df_wide,
                K=self.fc.K_weekly_to_hourly,
                period_days=self.fc.P_week,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        # 4. Daily
        if self.fc.K_daily_to_hourly > 0:
            df_wide, n_feats = append_fourier_features_matrix(
                df_wide,
                K=self.fc.K_daily_to_hourly,
                period_days=self.fc.P_day,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        return df_wide.sort_values('ds').reset_index(drop=True)

    @staticmethod
    def extract_XY(df_wide, H):
        """Extract X and Y from wide format DataFrame."""
        y_cols = sorted([c for c in df_wide.columns if c.startswith("y_")],
                        key=lambda s: int(s.split("_")[1]))
        x_cols = ["x_0"] + sorted([c for c in df_wide.columns if c.startswith("x_") and c != "x_0"],
                                  key=lambda s: int(s.split("_")[1]))

        X = df_wide[x_cols].to_numpy(dtype=np.float32)
        Y = df_wide[y_cols].to_numpy(dtype=np.float32)

        F_total = X.shape[1] - 1
        K = F_total // (2 * H) if F_total > 0 else 0

        harmonic_orders = sum(([k, k] for k in range(1, K + 1)), [])

        return X, Y, x_cols, y_cols, harmonic_orders

    @staticmethod
    def make_seq(X, Y, T=32, stride=1):
        """Create sequences for training."""
        N = len(X)
        if N < T:
            return X[None, ...], Y[None, ...]

        idx = list(range(0, N - T + 1, stride))
        X_seq = np.stack([X[i:i + T] for i in idx], axis=0).astype(np.float32)
        Y_seq = np.stack([Y[i:i + T] for i in idx], axis=0).astype(np.float32)
        return X_seq, Y_seq

    def predict_with_sequence_context(self, model, X_sequence, scaler_x=None, scaler_y=None):
        """
        Make predictions using full sequence context.

        Args:
            model: The RNN model (rnn_daily_base or rnn_hourly)
            X_sequence: Input sequence (T, F) or (B, T, F)
            scaler_x: Input scaler
            scaler_y: Output scaler

        Returns:
            Predictions for the last timestep in the sequence
        """
        device = self.tc.device

        # Ensure proper shape
        if X_sequence.ndim == 2:
            X_sequence = X_sequence[np.newaxis, ...]  # Add batch dimension

        B, T, F = X_sequence.shape

        # Normalize if scaler provided
        if scaler_x is not None:
            X_norm = np.zeros_like(X_sequence)
            for b in range(B):
                for t in range(T):
                    X_norm[b, t] = scaler_x.transform(X_sequence[b, t:t + 1])[0]
        else:
            X_norm = X_sequence

        # Convert to tensor
        X_tensor = torch.from_numpy(X_norm.astype(np.float32)).to(device)

        # Forward pass through model
        with torch.no_grad():
            model.eval()
            outputs, _, _ = model(X_tensor, return_hidden=True)

        # Get the last timestep's output
        final_output = outputs[:, -1, :].cpu().numpy()  # (B, H)

        # Denormalize if scaler provided
        if scaler_y is not None:
            final_output = scaler_y.inverse_transform(final_output)

        return final_output[0] if B == 1 else final_output

    class XYSeqDataset(Dataset):
        def __init__(self, X_seq, Y_seq):
            self.X = torch.from_numpy(X_seq).float()
            self.Y = torch.from_numpy(Y_seq).float()

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return self.X[i], self.Y[i]


# ============================================================================
# Training Approach: Teacher Forcing
# ============================================================================

class Approach1_TeacherForcing_2Stage:
    """
    Train each branch independently with ground truth inputs.
    """

    def __init__(self, downscaler: HierarchicalDownscaler_2Stage):
        self.ds = get_base_model(downscaler)
        self.ds_wrapped = downscaler
        self.tc = self.ds.tc
        self.fc = self.ds.fc

    def train_branch_1(self, df_yearly, df_daily):
        """Train Branch 1: Yearly → Daily."""
        print("\n" + "=" * 60)
        print("Training Branch 1: Yearly → Daily")
        print("=" * 60)

        df_wide = self.ds.prepare_yearly_to_daily_data(df_yearly, df_daily)
        X, Y, _, _, harmonic_orders = self.ds.extract_XY(df_wide, H=366)

        X_train, Y_train = X, Y
        X_test, Y_test = X, Y

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        self.ds.scalers['yearly']['sx'] = sx
        self.ds.scalers['yearly']['sy'] = sy

        Xs_train = sx.transform(X_train).astype(np.float32)
        Ys_train = sy.transform(Y_train).astype(np.float32)

        X_seq, Y_seq = self.ds.make_seq(Xs_train, Ys_train, T=self.tc.T_seq_yearly, stride=self.tc.stride)

        dataset = self.ds.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(dataset, batch_size=self.tc.batch_size, shuffle=True)

        model = self.ds.rnn_daily_base
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tc.lr)
        n_epochs = self.tc.base_epochs * self.tc.yearly_to_daily_multiplier

        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.tc.device), yb.to(self.tc.device)

                optimizer.zero_grad()
                o, _ = model(xb)

                data_loss = F.mse_loss(o, yb)
                reg_loss = model.reg_loss(
                    lambda0=self.tc.lambda0,
                    lambdaf=self.tc.lambdaf,
                    harmonic_orders=harmonic_orders
                )

                loss = data_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(loader):.6f}")

        print("Branch 1 training completed!")
        return X_test, Y_test

    def train_branch_2(self, df_daily, df_hourly):
        """Train Branch 2: Daily → Hourly."""
        print("\n" + "=" * 60)
        print("Training Branch 2: Daily → Hourly")
        print("=" * 60)

        df_wide = self.ds.prepare_daily_to_hourly_data(df_daily, df_hourly)
        X, Y, _, _, harmonic_orders = self.ds.extract_XY(df_wide, H=24)

        X_train, Y_train = X, Y
        X_test, Y_test = X, Y

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        self.ds.scalers['daily']['sx'] = sx
        self.ds.scalers['daily']['sy'] = sy

        Xs_train = sx.transform(X_train).astype(np.float32)
        Ys_train = sy.transform(Y_train).astype(np.float32)

        X_seq, Y_seq = self.ds.make_seq(Xs_train, Ys_train, T=self.tc.T_seq_daily, stride=self.tc.stride)

        dataset = self.ds.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(dataset, batch_size=self.tc.batch_size, shuffle=True)

        model = self.ds.rnn_hourly
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tc.lr)
        n_epochs = self.tc.base_epochs * self.tc.daily_to_hourly_multiplier

        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.tc.device), yb.to(self.tc.device)

                optimizer.zero_grad()
                o, _ = model(xb)

                data_loss = F.mse_loss(o, yb)
                reg_loss = model.reg_loss(
                    lambda0=self.tc.lambda0,
                    lambdaf=self.tc.lambdaf,
                    harmonic_orders=harmonic_orders
                )

                loss = data_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(loader):.6f}")

        print("Branch 2 training completed!")
        return X_test, Y_test

    def train(self, df_hourly):
        """Train all branches sequentially with teacher forcing."""
        print("\n" + "=" * 80)
        print("APPROACH 1: TEACHER FORCING (2-STAGE)")
        print("Training Yearly → Daily → Hourly")
        print("=" * 80)

        data_dict = self.ds.prepare_data(df_hourly)

        self.train_branch_1(data_dict['yearly'], data_dict['daily'])
        self.train_branch_2(data_dict['daily'], data_dict['hourly'])

        print("\n" + "=" * 80)
        print("APPROACH 1 TRAINING COMPLETED!")
        print("=" * 80)


class Approach2_CurriculumLearning_2Stage:
    """
    Approach 2: Scheduled sampling / curriculum learning for 2-stage model.
    Gradually mix ground truth and predicted inputs during training.
    """

    def __init__(self, downscaler: HierarchicalDownscaler_2Stage):
        self.ds = get_base_model(downscaler)
        self.ds_wrapped = downscaler
        self.tc = self.ds.tc
        self.fc = self.ds.fc

    def train_branch_1(self, df_yearly, df_daily):
        """Train Branch 1: Yearly → Daily (same as Approach 1)."""
        # Branch 1 has no upstream, so same as teacher forcing
        approach1 = Approach1_TeacherForcing_2Stage(self.ds_wrapped)
        return approach1.train_branch_1(df_yearly, df_daily)

    def train_branch_2_with_curriculum(self, df_yearly, df_daily, df_hourly):
        """Train Branch 2 with curriculum learning: mix true and predicted daily values."""
        print("\n" + "=" * 60)
        print("Training Branch 2: Daily → Hourly (with Curriculum Learning)")
        print("=" * 60)

        # Prepare ground truth data
        df_wide_daily = self.ds.prepare_daily_to_hourly_data(df_daily, df_hourly)
        X_true, Y, _, _, harmonic_orders = self.ds.extract_XY(df_wide_daily, H=24)

        # Split train/test
        X_train_true, Y_train = X_true[:-1], Y[:-1]

        # Standardize based on true data
        sx = StandardScaler().fit(X_train_true)
        sy = StandardScaler().fit(Y_train)
        self.ds.scalers['daily']['sx'] = sx
        self.ds.scalers['daily']['sy'] = sy

        # Prepare predicted daily values from Branch 1
        # We need to predict daily values for all days in training set
        df_wide_yearly = self.ds.prepare_yearly_to_daily_data(df_yearly, df_daily)
        X_yearly, Y_daily_true, _, _, _ = self.ds.extract_XY(df_wide_yearly, H=366)

        # Get predictions from Branch 1 for all years
        self.ds.rnn_daily_base.eval()
        device = self.tc.device

        # Normalize yearly inputs
        sx_yearly = self.ds.scalers['yearly']['sx']
        sy_yearly = self.ds.scalers['yearly']['sy']
        X_yearly_norm = sx_yearly.transform(X_yearly).astype(np.float32)

        # Predict daily values for all years
        X_yearly_tensor = torch.from_numpy(X_yearly_norm).unsqueeze(0).to(device)  # (1, N_years, F)
        with torch.no_grad():
            daily_pred_all, _ = self.ds.rnn_daily_base(X_yearly_tensor)  # (1, N_years, 366)
            daily_pred_all_np = daily_pred_all.squeeze(0).cpu().numpy()  # (N_years, 366)

        # Denormalize
        daily_pred_all_np = sy_yearly.inverse_transform(daily_pred_all_np)

        # Reconstruct predicted daily dataframe matching df_daily structure
        predicted_daily_rows = []
        for year_idx, (_, year_row) in enumerate(df_yearly.iterrows()):
            year = year_row['year']
            days_in_year = get_days_in_year(year)
            daily_preds_year = daily_pred_all_np[year_idx, :days_in_year]

            # Get dates for this year
            year_mask = df_daily['ds'].dt.year == year
            dates = df_daily[year_mask]['ds'].values

            if len(dates) == len(daily_preds_year):
                for date, pred_val in zip(dates, daily_preds_year):
                    predicted_daily_rows.append({
                        'ds': date,
                        'daily_sum': pred_val
                    })

        df_daily_predicted = pd.DataFrame(predicted_daily_rows)

        # Prepare predicted daily to hourly features
        df_wide_daily_pred = self.ds.prepare_daily_to_hourly_data(df_daily_predicted, df_hourly)
        X_pred, _, _, _, _ = self.ds.extract_XY(df_wide_daily_pred, H=24)
        X_train_pred = X_pred[:-1]

        # Normalize both true and predicted X
        Xs_train_true = sx.transform(X_train_true).astype(np.float32)
        Xs_train_pred = sx.transform(X_train_pred).astype(np.float32)
        Ys_train = sy.transform(Y_train).astype(np.float32)

        # Create sequences for both
        X_seq_true, Y_seq = self.ds.make_seq(Xs_train_true, Ys_train, T=self.tc.T_seq_daily, stride=self.tc.stride)
        X_seq_pred, _ = self.ds.make_seq(Xs_train_pred, Ys_train, T=self.tc.T_seq_daily, stride=self.tc.stride)

        # DataLoader
        dataset = self.ds.XYSeqDataset(X_seq_true, Y_seq)  # We'll manually mix in training loop
        loader = DataLoader(dataset, batch_size=self.tc.batch_size, shuffle=True)

        # Train
        model = self.ds.rnn_hourly
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tc.lr)
        n_epochs = self.tc.base_epochs * self.tc.daily_to_hourly_multiplier

        # Convert predicted sequences to tensor
        X_seq_pred_tensor = torch.from_numpy(X_seq_pred).float().to(device)

        model.train()
        for epoch in range(n_epochs):
            # Curriculum schedule: linearly decrease use of true data
            use_true_prob = self.tc.curriculum_start_prob - \
                            (epoch / n_epochs) * (self.tc.curriculum_start_prob - self.tc.curriculum_end_prob)

            total_loss = 0
            batch_idx = 0

            for xb_true, yb in loader:
                xb_true, yb = xb_true.to(device), yb.to(device)

                # Get corresponding predicted batch
                batch_size = xb_true.shape[0]
                start_idx = batch_idx * self.tc.batch_size
                end_idx = start_idx + batch_size

                if end_idx <= X_seq_pred_tensor.shape[0]:
                    xb_pred = X_seq_pred_tensor[start_idx:end_idx]

                    # Mix true and predicted inputs based on curriculum probability
                    if np.random.random() < use_true_prob:
                        # Teacher forcing: use true inputs
                        inputs = xb_true
                    else:
                        # Use predicted inputs
                        inputs = xb_pred
                else:
                    # Fallback to true inputs if predicted batch not available
                    inputs = xb_true

                optimizer.zero_grad()
                o, _ = model(inputs)

                data_loss = F.mse_loss(o, yb)
                reg_loss = model.reg_loss(
                    lambda0=self.tc.lambda0,
                    lambdaf=self.tc.lambdaf,
                    harmonic_orders=harmonic_orders
                )

                loss = data_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_idx += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(loader):.6f}, "
                      f"True prob: {use_true_prob:.2f}")

        print("Branch 2 training with curriculum learning completed!")

    def train(self, df_hourly):
        """Train all branches with curriculum learning."""
        print("\n" + "=" * 80)
        print("APPROACH 2: CURRICULUM LEARNING (2-STAGE)")
        print("Gradually mixing ground truth and predicted inputs")
        print("=" * 80)

        # Prepare hierarchical data
        data_dict = self.ds.prepare_data(df_hourly)

        # Train branches
        self.train_branch_1(data_dict['yearly'], data_dict['daily'])
        self.train_branch_2_with_curriculum(
            data_dict['yearly'],
            data_dict['daily'],
            data_dict['hourly']
        )

        print("\n" + "=" * 80)
        print("APPROACH 2 TRAINING COMPLETED!")
        print("=" * 80)

    def estimate_uncertainty(self, df_hourly):
        """
        Estimate uncertainty parameters (sigma matrices) after training.
        Returns sigma_stage1 and sigma_stage2.
        """
        print("\n" + "=" * 80)
        print("ESTIMATING UNCERTAINTY PARAMETERS")
        print("=" * 80)

        # Prepare data
        data_dict = self.ds.prepare_data(df_hourly)

        # Estimate sigma for both stages
        sigma_stage1 = self._estimate_sigma_stage1(data_dict)
        sigma_stage2 = self._estimate_sigma_stage2(data_dict)

        print("\n" + "=" * 80)
        print("UNCERTAINTY ESTIMATION COMPLETED!")
        print("=" * 80)

        return {
            'sigma_stage1': sigma_stage1,
            'sigma_stage2': sigma_stage2
        }

    def _estimate_sigma_stage1(self, data_dict):
        """
        Estimate Stage 1 (Yearly->Daily) covariance matrix from training residuals.
        Uses real yearly inputs to predict daily values.

        Returns:
            sigma_stage1: (366, 366) covariance matrix
        """
        print("\nEstimating Stage 1 (Yearly->Daily) uncertainty...")

        df_yearly = data_dict['yearly']
        df_daily = data_dict['daily']

        self.ds.rnn_daily_base.eval()

        residuals_stage1 = []

        with torch.no_grad():
            for year in sorted(df_yearly['year'].unique()):
                # Get predictor
                predictor = HierarchicalPredictor_2Stage(self.ds_wrapped)
                predictor.reset_hidden_states()

                # Get year data
                year_row = df_yearly[df_yearly['year'] == year].iloc[0]
                yearly_sum = year_row['yearly_sum']

                # Predict daily values using Stage 1
                try:
                    daily_pred = predictor._predict_daily(yearly_sum, year, historical_years=None)
                except:
                    continue

                # Get actual daily values
                daily_actual = df_daily[df_daily['year'] == year]['daily_sum'].values
                days_in_year = get_days_in_year(year)

                if len(daily_actual) < days_in_year:
                    continue

                daily_actual = daily_actual[:days_in_year]

                # Pad to 366 for consistent matrix size
                if days_in_year == 365:
                    daily_pred = np.pad(daily_pred, (0, 1), mode='edge')
                    daily_actual = np.pad(daily_actual, (0, 1), mode='edge')

                # Compute residual
                residual = daily_actual - daily_pred
                residuals_stage1.append(residual)

        # Stack and compute covariance
        residuals_stage1 = np.array(residuals_stage1)  # (n_years, 366)
        sigma_stage1 = np.cov(residuals_stage1, rowvar=False)  # (366, 366)

        print(f"  Stage 1 sigma estimated from {len(residuals_stage1)} years")
        print(f"  Sigma shape: {sigma_stage1.shape}")
        print(f"  Mean diagonal variance: {np.mean(np.diag(sigma_stage1)):.4f}")

        return sigma_stage1

    def _estimate_sigma_stage2(self, data_dict):
        """
        Estimate Stage 2 (Daily->Hourly) covariance matrix from training residuals.
        CRITICAL: Uses Stage 1 PREDICTIONS (not real daily values) as input.

        Returns:
            sigma_stage2: (24, 24) covariance matrix
        """
        print("\nEstimating Stage 2 (Daily->Hourly) uncertainty...")

        df_yearly = data_dict['yearly']
        df_daily = data_dict['daily']
        df_hourly = data_dict['hourly']

        self.ds.rnn_daily_base.eval()
        self.ds.rnn_hourly.eval()

        residuals_stage2 = []

        with torch.no_grad():
            for year in sorted(df_yearly['year'].unique()):
                # Get predictor
                predictor = HierarchicalPredictor_2Stage(self.ds_wrapped)
                predictor.reset_hidden_states()

                # Get year data
                year_row = df_yearly[df_yearly['year'] == year].iloc[0]
                yearly_sum = year_row['yearly_sum']

                # Step 1: Get Stage 1 prediction for this year
                try:
                    daily_pred = predictor._predict_daily(yearly_sum, year, historical_years=None)
                except:
                    continue

                days_in_year = get_days_in_year(year)

                # Step 2: Use predicted daily values to predict hourly
                historical_days = []
                for day_idx in range(days_in_year):
                    start_date = pd.Timestamp(f'{year}-01-01')
                    current_date = start_date + pd.Timedelta(days=day_idx)

                    # Get actual hourly values
                    hourly_actual = df_hourly[df_hourly['date'] == current_date.date()]['y'].values

                    if len(hourly_actual) != 24:
                        continue

                    # Predict hourly using Stage 1's predicted daily value
                    try:
                        hourly_pred = predictor._predict_hourly(
                            daily_sum=daily_pred[day_idx],  # Use PREDICTED daily
                            year=year,
                            day_idx=day_idx,
                            historical_days=historical_days.copy() if len(historical_days) > 0 else None
                        )
                    except:
                        continue

                    # Compute residual
                    residual = hourly_actual - hourly_pred  # (24,)
                    residuals_stage2.append(residual)

                    # Update historical
                    historical_days.append({
                        'date': current_date,
                        'daily_sum': daily_pred[day_idx]
                    })

        # Stack and compute covariance
        residuals_stage2 = np.array(residuals_stage2)  # (n_days_total, 24)
        sigma_stage2 = np.cov(residuals_stage2, rowvar=False)  # (24, 24)

        print(f"  Stage 2 sigma estimated from {len(residuals_stage2)} days")
        print(f"  Sigma shape: {sigma_stage2.shape}")
        print(f"  Mean diagonal variance: {np.mean(np.diag(sigma_stage2)):.4f}")

        return sigma_stage2


class Approach3_EndToEnd_2Stage:
    """
    Approach 3: End-to-end training of 2-stage model.
    Both stages are trained together with full gradient flow from hourly loss back through both stages.
    """

    def __init__(self, downscaler: HierarchicalDownscaler_2Stage):
        self.ds = get_base_model(downscaler)
        self.ds_wrapped = downscaler
        self.tc = self.ds.tc
        self.fc = self.ds.fc

    def train(self, df_hourly):
        """Train both stages end-to-end with gradient flow through the entire pipeline."""
        print("\n" + "=" * 80)
        print("APPROACH 3: END-TO-END TRAINING (2-STAGE)")
        print("Training Yearly → Daily → Hourly with full gradient flow")
        print("=" * 80)

        # Prepare hierarchical data
        data_dict = self.ds.prepare_data(df_hourly)
        df_yearly = data_dict['yearly']
        df_daily = data_dict['daily']
        df_hourly_data = data_dict['hourly']

        # Prepare yearly to daily data
        df_wide_yearly = self.ds.prepare_yearly_to_daily_data(df_yearly, df_daily)
        X_yearly, Y_daily, _, _, harmonic_orders_stage1 = self.ds.extract_XY(df_wide_yearly, H=366)

        # Prepare daily to hourly data
        df_wide_daily = self.ds.prepare_daily_to_hourly_data(df_daily, df_hourly_data)
        X_daily, Y_hourly, _, _, harmonic_orders_stage2 = self.ds.extract_XY(df_wide_daily, H=24)

        # Split train/test
        X_yearly_train = X_yearly[:-1]
        Y_daily_train = Y_daily[:-1]
        X_daily_train = X_daily[:-1]
        Y_hourly_train = Y_hourly[:-1]

        # Setup scalers
        sx_yearly = StandardScaler().fit(X_yearly_train)
        sy_yearly = StandardScaler().fit(Y_daily_train)
        sx_daily = StandardScaler().fit(X_daily_train)
        sy_hourly = StandardScaler().fit(Y_hourly_train)

        self.ds.scalers['yearly']['sx'] = sx_yearly
        self.ds.scalers['yearly']['sy'] = sy_yearly
        self.ds.scalers['daily']['sx'] = sx_daily
        self.ds.scalers['daily']['sy'] = sy_hourly

        # Normalize data
        X_yearly_train_norm = sx_yearly.transform(X_yearly_train).astype(np.float32)
        Y_daily_train_norm = sy_yearly.transform(Y_daily_train).astype(np.float32)
        Y_hourly_train_norm = sy_hourly.transform(Y_hourly_train).astype(np.float32)

        # Create sequences for Stage 1 (yearly)
        X_seq_yearly, Y_seq_daily = self.ds.make_seq(
            X_yearly_train_norm, Y_daily_train_norm,
            T=self.tc.T_seq_yearly,
            stride=self.tc.stride
        )

        # Create mapping from yearly sequence index to daily data indices
        # Each year in the sequence corresponds to 365/366 days
        # We need to track which days correspond to which year sequence
        year_to_days_map = []
        cumsum_days = 0
        for year_idx in range(len(df_yearly) - 1):
            year = df_yearly.iloc[year_idx]['year']
            days_in_year = get_days_in_year(year)
            year_to_days_map.append((cumsum_days, cumsum_days + days_in_year))
            cumsum_days += days_in_year

        # Setup models and optimizer
        device = self.tc.device
        model_stage1 = self.ds.rnn_daily_base
        model_stage2 = self.ds.rnn_hourly

        # Combine parameters from both stages
        all_parameters = list(model_stage1.parameters()) + list(model_stage2.parameters())
        optimizer = torch.optim.Adam(all_parameters, lr=self.tc.lr)

        # Training parameters
        n_epochs = self.tc.base_epochs

        print(f"\nTraining for {n_epochs} epochs")
        print(f"Yearly sequences: {X_seq_yearly.shape}")
        print(f"Device: {device}")

        # Training loop
        model_stage1.train()
        model_stage2.train()

        for epoch in range(n_epochs):
            total_loss = 0
            total_loss_stage1 = 0
            total_loss_stage2 = 0
            n_batches = 0

            # Process each yearly sequence
            for seq_idx in range(len(X_seq_yearly)):
                # Get yearly sequence
                x_yearly_seq = torch.from_numpy(X_seq_yearly[seq_idx:seq_idx + 1]).float().to(device)  # (1, T_seq, F)
                y_daily_seq = torch.from_numpy(Y_seq_daily[seq_idx:seq_idx + 1]).float().to(device)  # (1, T_seq, 366)

                # Forward through Stage 1: Yearly → Daily
                daily_pred, _ = model_stage1(x_yearly_seq)  # (1, T_seq, 366)

                # Stage 1 loss
                loss_stage1 = F.mse_loss(daily_pred, y_daily_seq)
                reg_loss_stage1 = model_stage1.reg_loss(
                    lambda0=self.tc.lambda0,
                    lambdaf=self.tc.lambdaf,
                    harmonic_orders=harmonic_orders_stage1
                )

                # Get the last year's daily predictions for Stage 2
                # Shape: (1, 366) → need to select actual days and prepare for Stage 2
                last_year_idx = seq_idx + self.tc.T_seq_yearly - 1

                if last_year_idx >= len(year_to_days_map):
                    continue

                day_start, day_end = year_to_days_map[last_year_idx]
                n_days = day_end - day_start

                # Get daily predictions for the last year in sequence (in normalized space)
                daily_pred_last = daily_pred[0, -1, :n_days]  # (n_days,)

                # Denormalize to original scale for Stage 2 input
                daily_pred_last_np = daily_pred_last.detach().cpu().numpy()
                daily_pred_padded = np.pad(daily_pred_last_np, (0, 366 - n_days), mode='edge') if n_days < 366 else daily_pred_last_np
                daily_pred_last_denorm = sy_yearly.inverse_transform(daily_pred_padded.reshape(1, -1)).flatten()[:n_days]

                # Prepare Stage 2 input features using predicted daily values
                # Build features for these days
                year = df_yearly.iloc[last_year_idx]['year']
                start_date = pd.Timestamp(f'{year}-01-01')

                # Create dataframe with predicted daily sums
                predicted_days_data = []
                for day_idx in range(n_days):
                    current_date = start_date + pd.Timedelta(days=day_idx)
                    predicted_days_data.append({
                        'ds': current_date,
                        'daily_sum': daily_pred_last_denorm[day_idx]
                    })

                df_predicted_days = pd.DataFrame(predicted_days_data)

                # Get hourly data for these days
                year_mask = df_hourly_data['ds'].dt.year == year
                df_hourly_year = df_hourly_data[year_mask]

                # Prepare Stage 2 features
                df_wide_pred = self.ds.prepare_daily_to_hourly_data(df_predicted_days, df_hourly_year)
                X_daily_pred, Y_hourly_true, _, _, _ = self.ds.extract_XY(df_wide_pred, H=24)

                if len(X_daily_pred) == 0:
                    continue

                # Normalize Stage 2 inputs
                X_daily_pred_norm = sx_daily.transform(X_daily_pred).astype(np.float32)
                Y_hourly_true_norm = sy_hourly.transform(Y_hourly_true).astype(np.float32)

                # Create sequences for Stage 2
                X_seq_daily_pred, Y_seq_hourly = self.ds.make_seq(
                    X_daily_pred_norm, Y_hourly_true_norm,
                    T=self.tc.T_seq_daily,
                    stride=1
                )

                if len(X_seq_daily_pred) == 0:
                    continue

                # Sample a batch of daily sequences for Stage 2
                batch_size = min(self.tc.batch_size, len(X_seq_daily_pred))
                batch_indices = np.random.choice(len(X_seq_daily_pred), size=batch_size, replace=False)

                total_loss_stage1 += loss_stage1.item()

                for batch_idx in batch_indices:
                    optimizer.zero_grad()
                    x_daily_seq = torch.from_numpy(X_seq_daily_pred[batch_idx:batch_idx + 1]).float().to(device)
                    y_hourly_seq = torch.from_numpy(Y_seq_hourly[batch_idx:batch_idx + 1]).float().to(device)

                    # Forward through Stage 2: Daily → Hourly
                    hourly_pred, _ = model_stage2(x_daily_seq)  # (1, T_seq_daily, 24)

                    # Stage 2 loss
                    loss_stage2 = F.mse_loss(hourly_pred, y_hourly_seq)
                    reg_loss_stage2 = model_stage2.reg_loss(
                        lambda0=self.tc.lambda0,
                        lambdaf=self.tc.lambdaf,
                        harmonic_orders=harmonic_orders_stage2
                    )

                    # Combined loss
                    loss = loss_stage1 + reg_loss_stage1 + loss_stage2 + reg_loss_stage2

                    # Backward and optimize
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_loss_stage2 += loss_stage2.item()
                    n_batches += 1

            if n_batches > 0:
                avg_loss = total_loss / n_batches
                avg_loss_stage1 = total_loss_stage1 / n_batches
                avg_loss_stage2 = total_loss_stage2 / n_batches

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{n_epochs}, "
                          f"Total Loss: {avg_loss:.6f}, "
                          f"Stage1: {avg_loss_stage1:.6f}, "
                          f"Stage2: {avg_loss_stage2:.6f}")

        print("\n" + "=" * 80)
        print("END-TO-END TRAINING COMPLETED!")
        print("=" * 80)


class NoiseAugmentedTraining:
    """
    Train Stage 2 (Daily → Hourly) with noise augmentation to handle Stage 1 errors.

    Key idea: Add noise to daily inputs during training to simulate the errors
    that Stage 1 will produce at test time.
    """

    def __init__(self, downscaler, stage1_error_std=None):
        """
        Args:
            downscaler: HierarchicalDownscaler_2Stage instance
            stage1_error_std: Standard deviation of Stage 1 errors (float or dict)
                             If None, will be estimated from data
                             If float, uses constant noise
                             If dict, can specify per-parameter noise
        """
        from Hierarchical_RNN_2stage_uncertainty import get_base_model

        self.ds = get_base_model(downscaler)
        self.ds_wrapped = downscaler
        self.tc = self.ds.tc
        self.fc = self.ds.fc

        # Stage 1 error distribution
        self.stage1_error_std = stage1_error_std

    def estimate_stage1_errors(self, df_hourly, n_validation_years=2):
        """
        Estimate Stage 1 error distribution from validation data.

        Args:
            df_hourly: Full hourly dataframe
            n_validation_years: Number of recent years to use for error estimation

        Returns:
            dict with error statistics
        """
        print("\n" + "=" * 70)
        print("ESTIMATING STAGE 1 ERROR DISTRIBUTION")
        print("=" * 70)

        # Prepare data
        data_dict = self.ds.prepare_data(df_hourly)
        df_yearly = data_dict['yearly']
        df_daily = data_dict['daily']

        # Use last n years for validation
        years = sorted(df_yearly['year'].unique())
        validation_years = years[-n_validation_years:]

        print(f"Using validation years: {validation_years}")

        # Train Stage 1 on earlier data
        train_mask = ~df_yearly['year'].isin(validation_years)
        df_yearly_train = df_yearly[train_mask]
        df_daily_train = df_daily[df_daily['year'].isin(df_yearly_train['year'])]

        print(f"Training Stage 1 on {len(df_yearly_train)} years...")
        self._train_stage1_only(df_yearly_train, df_daily_train)

        # Get predictions on validation years
        print(f"Generating Stage 1 predictions on validation years...")
        errors_abs = []
        errors_rel = []

        for year in validation_years:
            year_data = df_yearly[df_yearly['year'] == year].iloc[0]
            yearly_sum = year_data['yearly_sum']

            # Get prediction
            from Hierarchical_RNN_2stage_uncertainty import HierarchicalPredictor_2Stage
            predictor = HierarchicalPredictor_2Stage(self.ds_wrapped)

            # Get historical years for context
            historical = df_yearly[df_yearly['year'] < year]
            historical_years = [
                {'year': row['year'], 'yearly_sum': row['yearly_sum']}
                for _, row in historical.iterrows()
            ]

            daily_pred = predictor._predict_daily(yearly_sum, year, historical_years)

            # Get ground truth
            daily_true = df_daily[df_daily['year'] == year]['daily_sum'].values

            # Compute errors
            if len(daily_pred) == len(daily_true):
                abs_errors = np.abs(daily_pred - daily_true)
                rel_errors = abs_errors / (daily_true + 1e-8)

                errors_abs.extend(abs_errors)
                errors_rel.extend(rel_errors)

        errors_abs = np.array(errors_abs)
        errors_rel = np.array(errors_rel)

        # Compute statistics
        error_stats = {
            'mean_abs': np.mean(errors_abs),
            'std_abs': np.std(errors_abs),
            'median_abs': np.median(errors_abs),
            'q25_abs': np.percentile(errors_abs, 25),
            'q75_abs': np.percentile(errors_abs, 75),
            'mean_rel': np.mean(errors_rel),
            'std_rel': np.std(errors_rel),
        }

        print("\n" + "=" * 70)
        print("STAGE 1 ERROR STATISTICS")
        print("=" * 70)
        print(f"Mean absolute error: {error_stats['mean_abs']:.2f}")
        print(f"Std absolute error:  {error_stats['std_abs']:.2f}")
        print(f"Median absolute error: {error_stats['median_abs']:.2f}")
        print(f"Q25-Q75 range: [{error_stats['q25_abs']:.2f}, {error_stats['q75_abs']:.2f}]")
        print(f"Mean relative error: {error_stats['mean_rel']:.2%}")
        print(f"Std relative error:  {error_stats['std_rel']:.2%}")
        print("=" * 70)

        return error_stats

    def _train_stage1_only(self, df_yearly, df_daily):
        """Quick training of Stage 1 for error estimation."""
        from Hierarchical_RNN_2stage_uncertainty import HierarchicalDownscaler_2Stage

        # Prepare data
        df_wide = self.ds.prepare_yearly_to_daily_data(df_yearly, df_daily)
        X, Y, _, _, harmonic_orders = self.ds.extract_XY(df_wide, H=366)

        # Simple train/test split
        n_train = max(1, len(X) - 1)
        X_train, Y_train = X[:n_train], Y[:n_train]

        # Standardize
        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        self.ds.scalers['yearly']['sx'] = sx
        self.ds.scalers['yearly']['sy'] = sy

        Xs_train = sx.transform(X_train).astype(np.float32)
        Ys_train = sy.transform(Y_train).astype(np.float32)

        # Create sequences
        X_seq, Y_seq = self.ds.make_seq(Xs_train, Ys_train, T=self.tc.T_seq_yearly, stride=self.tc.stride)

        # DataLoader
        dataset = self.ds.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(dataset, batch_size=min(self.tc.batch_size, len(X_seq)), shuffle=True)

        # Quick training
        model = self.ds.rnn_daily_base
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tc.lr)
        n_epochs = 50  # Quick training

        model.train()
        for epoch in range(n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.tc.device), yb.to(self.tc.device)
                optimizer.zero_grad()
                o, _ = model(xb)
                loss = F.mse_loss(o, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    def train_stage2_with_noise(self, df_daily, df_hourly,
                                noise_schedule='curriculum',
                                noise_multiplier=1.0,
                                min_noise_ratio=0.0,
                                max_noise_ratio=1.5):
        """
        Train Stage 2 (Daily → Hourly) with noise augmentation.

        Args:
            df_daily: Daily dataframe
            df_hourly: Hourly dataframe
            noise_schedule: 'constant', 'curriculum', or 'random'
                - constant: Fixed noise throughout training
                - curriculum: Start low, increase noise over time
                - random: Random noise level per batch
            noise_multiplier: Scale factor for base noise level
            min_noise_ratio: Minimum noise as ratio of error_std (for curriculum)
            max_noise_ratio: Maximum noise as ratio of error_std (for curriculum)
        """
        print("\n" + "=" * 70)
        print("TRAINING STAGE 2 WITH NOISE AUGMENTATION")
        print("=" * 70)
        print(f"Noise schedule: {noise_schedule}")
        print(f"Noise multiplier: {noise_multiplier}")

        if self.stage1_error_std is None:
            print("WARNING: stage1_error_std not provided. Using default value of 500.0")
            error_std = 500.0
        elif isinstance(self.stage1_error_std, dict):
            error_std = self.stage1_error_std['std_abs']
        else:
            error_std = self.stage1_error_std

        print(f"Base error std: {error_std:.2f}")
        print("=" * 70)

        # Prepare data
        df_wide = self.ds.prepare_daily_to_hourly_data(df_daily, df_hourly)
        X, Y, _, _, harmonic_orders = self.ds.extract_XY(df_wide, H=24)

        # Split train/test
        X_train, Y_train = X, Y
        X_test, Y_test = X[-100:], Y[-100:]

        # Standardize
        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        self.ds.scalers['daily']['sx'] = sx
        self.ds.scalers['daily']['sy'] = sy

        Xs_train = sx.transform(X_train).astype(np.float32)
        Ys_train = sy.transform(Y_train).astype(np.float32)

        # Create sequences
        X_seq, Y_seq = self.ds.make_seq(Xs_train, Ys_train, T=self.tc.T_seq_daily, stride=self.tc.stride)

        print(f"\nTraining data shape: X_seq={X_seq.shape}, Y_seq={Y_seq.shape}")

        # DataLoader
        dataset = self.ds.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(dataset, batch_size=self.tc.batch_size, shuffle=True)

        # Train
        model = self.ds.rnn_hourly
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tc.lr)
        n_epochs = self.tc.base_epochs * self.tc.daily_to_hourly_multiplier

        print(f"Training for {n_epochs} epochs...")
        print("=" * 70)

        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            total_data_loss = 0
            total_reg_loss = 0
            n_batches = 0

            # Determine noise level for this epoch
            if noise_schedule == 'constant':
                noise_ratio = 1.0
            elif noise_schedule == 'curriculum':
                # Start with min_noise_ratio, linearly increase to max_noise_ratio
                progress = epoch / n_epochs
                noise_ratio = min_noise_ratio + (max_noise_ratio - min_noise_ratio) * progress
            else:  # random
                noise_ratio = np.random.uniform(min_noise_ratio, max_noise_ratio)

            current_noise_std = error_std * noise_ratio * noise_multiplier

            for xb, yb in loader:
                xb, yb = xb.to(self.tc.device), yb.to(self.tc.device)

                # ADD NOISE TO INPUT
                # x_0 is the daily sum at the first feature dimension
                xb_noisy = xb.clone()

                # Generate noise
                noise = torch.randn_like(xb[:, :, 0:1], device=self.tc.device) * current_noise_std

                # Transform noise to normalized space (since data is normalized)
                # We need to scale the noise by the input scaler
                noise_normalized = noise / sx.scale_[0]  # Scale noise appropriately

                # Add noise to the daily aggregate (first feature)
                xb_noisy[:, :, 0:1] = xb[:, :, 0:1] + noise_normalized

                # Forward pass with noisy input
                optimizer.zero_grad()
                o, _ = model(xb_noisy)

                # Compute loss
                data_loss = F.mse_loss(o, yb)
                reg_loss = model.reg_loss(
                    lambda0=self.tc.lambda0,
                    lambdaf=self.tc.lambdaf,
                    harmonic_orders=harmonic_orders
                )

                loss = data_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_data_loss += data_loss.item()
                total_reg_loss += reg_loss.item()
                n_batches += 1

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / n_batches
                avg_data = total_data_loss / n_batches
                avg_reg = total_reg_loss / n_batches
                print(f"Epoch {epoch + 1}/{n_epochs} | "
                      f"Loss: {avg_loss:.6f} (Data: {avg_data:.6f}, Reg: {avg_reg:.6f}) | "
                      f"Noise std: {current_noise_std:.2f} (ratio: {noise_ratio:.2f})")

        print("\n" + "=" * 70)
        print("STAGE 2 TRAINING WITH NOISE AUGMENTATION COMPLETED!")
        print("=" * 70)

        return X_test, Y_test

    def train(self, df_hourly, estimate_errors=True, n_validation_years=2):
        """
        Complete training pipeline with noise augmentation.

        Args:
            df_hourly: Full hourly dataframe
            estimate_errors: If True, estimate Stage 1 errors before training Stage 2
            n_validation_years: Number of years to use for error estimation
        """
        print("\n" + "=" * 80)
        print("NOISE AUGMENTED TRAINING - 2 STAGE MODEL")
        print("=" * 80)

        # Prepare data
        data_dict = self.ds.prepare_data(df_hourly)

        # Step 1: Train Stage 1 (Yearly → Daily)
        print("\n" + "=" * 70)
        print("STEP 1: TRAINING STAGE 1 (YEARLY → DAILY)")
        print("=" * 70)

        df_wide = self.ds.prepare_yearly_to_daily_data(data_dict['yearly'], data_dict['daily'])
        X, Y, _, _, harmonic_orders = self.ds.extract_XY(df_wide, H=366)

        X_train, Y_train = X[:-1], Y[:-1]

        sx = StandardScaler().fit(X_train)
        sy = StandardScaler().fit(Y_train)
        self.ds.scalers['yearly']['sx'] = sx
        self.ds.scalers['yearly']['sy'] = sy

        Xs_train = sx.transform(X_train).astype(np.float32)
        Ys_train = sy.transform(Y_train).astype(np.float32)

        X_seq, Y_seq = self.ds.make_seq(Xs_train, Ys_train, T=self.tc.T_seq_yearly, stride=self.tc.stride)

        dataset = self.ds.XYSeqDataset(X_seq, Y_seq)
        loader = DataLoader(dataset, batch_size=self.tc.batch_size, shuffle=True)

        model = self.ds.rnn_daily_base
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tc.lr)
        n_epochs = self.tc.base_epochs * self.tc.yearly_to_daily_multiplier

        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.tc.device), yb.to(self.tc.device)
                optimizer.zero_grad()
                o, _ = model(xb)

                data_loss = F.mse_loss(o, yb)
                reg_loss = model.reg_loss(
                    lambda0=self.tc.lambda0,
                    lambdaf=self.tc.lambdaf,
                    harmonic_orders=harmonic_orders
                )

                loss = data_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(loader):.6f}")

        print("Stage 1 training completed!")

        # Step 2: Estimate Stage 1 errors (if requested)
        if estimate_errors:
            error_stats = self.estimate_stage1_errors(df_hourly, n_validation_years)
            self.stage1_error_std = error_stats

        # Step 3: Train Stage 2 with noise augmentation
        self.train_stage2_with_noise(
            data_dict['daily'],
            data_dict['hourly'],
            noise_schedule='curriculum',
            noise_multiplier=1.0,
            min_noise_ratio=0.3,
            max_noise_ratio=1.5
        )

        print("\n" + "=" * 80)
        print("COMPLETE TRAINING FINISHED!")
        print("=" * 80)


# ============================================================================
# Inference / Prediction
# ============================================================================

class HierarchicalPredictor_2Stage:
    """Inference engine for 2-stage hierarchical downscaling."""

    def __init__(self, downscaler: HierarchicalDownscaler_2Stage):
        self.ds = get_base_model(downscaler)
        self.ds_wrapped = downscaler
        self.hidden_state_cache = {
            'yearly': None,
            'daily': None
        }

    def reset_hidden_states(self):
        """Reset cached hidden states for new sequence."""
        self.hidden_state_cache = {
            'yearly': None,
            'daily': None
        }

    @torch.no_grad()
    def predict_yearly_to_hourly(self, yearly_sum: float, year: int,
                                 historical_years=None, previous_year_daily=None,
                                 reset_hidden=True):
        """
        Predict all hourly values in a single forward pass.
        Similar to RNN_fourier_RNN.forecast_knownX approach.

        Args:
            yearly_sum: Target year's sum to downscale
            year: Target year
            historical_years: List of dicts with {'year': year, 'yearly_sum': sum} for context
            previous_year_daily: Last N days from previous year for daily context
            reset_hidden: Whether to reset hidden states (default True for each new year)
        """
        if reset_hidden:
            self.reset_hidden_states()

        self.ds.rnn_daily_base.eval()
        self.ds.rnn_hourly.eval()

        device = self.ds.tc.device

        # Step 1: Get daily predictions for the year
        daily_values = self._predict_daily(yearly_sum, year, historical_years)
        days_in_year = len(daily_values)

        # Step 2: Create one long sequence for the entire year
        # Start with T_seq_daily-1 days of context (from previous year or synthetic)
        start_date = pd.Timestamp(f'{year}-01-01')

        if previous_year_daily is not None and len(previous_year_daily) >= self.ds.tc.T_seq_daily - 1:
            # Use actual previous year data
            context_daily = previous_year_daily[-(self.ds.tc.T_seq_daily - 1):]
        else:
            # Create synthetic context
            context_daily = []
            for i in range(self.ds.tc.T_seq_daily - 1, 0, -1):
                context_daily.append({
                    'date': start_date - pd.Timedelta(days=i),
                    'daily_sum': daily_values[0] * (0.9 + 0.1 * np.random.random())  # Add variation
                })

        # Add all daily values for the year
        all_daily = context_daily.copy()
        for day_idx, daily_sum in enumerate(daily_values):
            all_daily.append({
                'date': start_date + pd.Timedelta(days=day_idx),
                'daily_sum': daily_sum
            })

        # Create DataFrame for the entire sequence
        df_seq = []
        for d_data in all_daily:
            df_seq.append({
                'ds': d_data['date'],
                'date': d_data['date'].date() if hasattr(d_data['date'], 'date') else d_data['date'],
                'x_0': d_data['daily_sum']
            })

        df_full = pd.DataFrame(df_seq)

        # Add Fourier features - STANDARD ORDER: yearly, monthly, weekly, daily
        start_idx = 1

        # 1. Yearly features
        if self.ds.fc.K_yearly_to_hourly > 0:
            df_full, n_feats = append_fourier_features_matrix(
                df_full,
                K=self.ds.fc.K_yearly_to_hourly,
                period_days=self.ds.fc.P_year,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        # 2. Monthly features
        if self.ds.fc.K_monthly_to_hourly > 0:
            df_full, n_feats = append_fourier_features_matrix(
                df_full,
                K=self.ds.fc.K_monthly_to_hourly,
                period_days=self.ds.fc.P_month,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        # 3. Weekly features
        if self.ds.fc.K_weekly_to_hourly > 0:
            df_full, n_feats = append_fourier_features_matrix(
                df_full,
                K=self.ds.fc.K_weekly_to_hourly,
                period_days=self.ds.fc.P_week,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        # 4. Daily features
        if self.ds.fc.K_daily_to_hourly > 0:
            df_full, n_feats = append_fourier_features_matrix(
                df_full,
                K=self.ds.fc.K_daily_to_hourly,
                period_days=self.ds.fc.P_day,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += n_feats

        # Extract features
        x_cols = ["x_0"] + sorted([c for c in df_full.columns if c.startswith("x_") and c != "x_0"],
                                  key=lambda s: int(s.split("_")[1]))
        X = df_full[x_cols].to_numpy(dtype=np.float32)

        # Split into historical context and future to predict
        X_hist = X[:self.ds.tc.T_seq_daily - 1]  # Context days
        X_future = X[self.ds.tc.T_seq_daily - 1:]  # Days to predict

        # Normalize
        sx = self.ds.scalers['daily']['sx']
        sy = self.ds.scalers['daily']['sy']

        if sx is not None:
            X_hist_norm = sx.transform(X_hist).astype(np.float32) if len(X_hist) > 0 else X_hist
            X_future_norm = sx.transform(X_future).astype(np.float32)
        else:
            X_hist_norm = X_hist
            X_future_norm = X_future

        # Concatenate historical and future
        X_full = np.concatenate([X_hist_norm, X_future_norm], axis=0) if len(X_hist_norm) > 0 else X_future_norm

        # Convert to tensor and add batch dimension
        X_tensor = torch.from_numpy(X_full).unsqueeze(0).to(device)  # (1, T_seq_daily-1+days_in_year, F)

        # Single forward pass through the model
        hourly_pred_all, _ = self.ds.rnn_hourly(X_tensor)  # (1, T_seq_daily-1+days_in_year, 24)

        # Extract predictions for the year (skip the context days)
        # We want predictions starting from index (T_seq_daily-1)
        # These correspond to the actual days of the year
        if len(X_hist) > 0:
            hourly_year = hourly_pred_all[0, len(X_hist):, :].cpu().numpy()  # (days_in_year, 24)
        else:
            hourly_year = hourly_pred_all[0, :, :].cpu().numpy()  # (days_in_year, 24)

        # Verify shape
        assert hourly_year.shape[0] == days_in_year, f"Expected {days_in_year} days, got {hourly_year.shape[0]}"

        # Denormalize
        if sy is not None:
            hourly_year_denorm = np.zeros_like(hourly_year)
            for i in range(days_in_year):
                hourly_year_denorm[i] = sy.inverse_transform(hourly_year[i:i + 1])[0]
            hourly_year = hourly_year_denorm

        # Flatten to get all hourly values for the year
        return hourly_year.flatten()

    def _predict_daily(self, yearly_sum: float, year: int, historical_years=None):
        """Predict daily values from yearly sum with sequence context."""
        device = self.ds.tc.device

        # Create sequence of years for context
        if historical_years is not None and len(historical_years) > 0:
            # Use actual historical years if provided
            years_to_use = historical_years + [{'year': year, 'yearly_sum': yearly_sum}]
            # Take last T_seq_yearly years for sequence
            years_to_use = years_to_use[-self.ds.tc.T_seq_yearly:]
        else:
            # For the first prediction, create minimal sequence
            years_to_use = [{'year': year, 'yearly_sum': yearly_sum}] * self.ds.tc.T_seq_yearly

        # Ensure we have exactly T_seq_yearly timesteps
        while len(years_to_use) < self.ds.tc.T_seq_yearly:
            years_to_use.insert(0, years_to_use[0].copy())

        # Build dataframe with sequence
        df_seq = []
        for y_data in years_to_use:
            dummy_date = pd.Timestamp(f"{y_data['year']}-01-01")
            df_seq.append({
                'ds': dummy_date,
                'year': y_data['year'],
                'x_0': y_data['yearly_sum']
            })

        df_dummy = pd.DataFrame(df_seq)

        # Add Fourier features - STANDARD ORDER: yearly, monthly, weekly
        start_idx = 1

        # 1. Yearly
        if self.ds.fc.K_yearly_to_daily > 0:
            df_dummy, _ = append_fourier_features_matrix(
                df_dummy,
                K=self.ds.fc.K_yearly_to_daily,
                period_days=self.ds.fc.P_year,
                H=366,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += 2 * self.ds.fc.K_yearly_to_daily * 366

        # 2. Monthly
        if self.ds.fc.K_monthly_to_daily > 0:
            df_dummy, _ = append_fourier_features_matrix(
                df_dummy,
                K=self.ds.fc.K_monthly_to_daily,
                period_days=self.ds.fc.P_month,
                H=366,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += 2 * self.ds.fc.K_monthly_to_daily * 366

        # 3. Weekly
        if self.ds.fc.K_weekly_to_daily > 0:
            df_dummy, _ = append_fourier_features_matrix(
                df_dummy,
                K=self.ds.fc.K_weekly_to_daily,
                period_days=self.ds.fc.P_week,
                H=366,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += 2 * self.ds.fc.K_weekly_to_daily * 366

        # Extract features
        x_cols = ["x_0"] + sorted([c for c in df_dummy.columns if c.startswith("x_") and c != "x_0"],
                                  key=lambda s: int(s.split("_")[1]))
        X = df_dummy[x_cols].to_numpy(dtype=np.float32)

        # Normalize
        sx = self.ds.scalers['yearly']['sx']
        sy = self.ds.scalers['yearly']['sy']

        if sx is not None:
            X_norm = sx.transform(X).astype(np.float32)
        else:
            X_norm = X

        # Predict 366 days - now with proper sequence shape
        X_tensor = torch.from_numpy(X_norm).unsqueeze(0).to(device)  # (1, T_seq_yearly, F)

        # CRITICAL: We want the prediction for the LAST year in the sequence
        # The model outputs (1, T_seq_yearly, 366)
        # We want the prediction for the target year, which is the last timestep

        if self.hidden_state_cache['yearly'] is not None and hasattr(self.ds.rnn_daily_base, 'forward'):
            daily_pred_366, _, self.hidden_state_cache['yearly'] = self.ds.rnn_daily_base(
                X_tensor, z0=self.hidden_state_cache['yearly'], return_hidden=True
            )
        else:
            daily_pred_366, _ = self.ds.rnn_daily_base(X_tensor)

        # CORRECT EXTRACTION: Take the LAST timestep (the target year)
        # Shape: (1, T_seq_yearly, 366) -> (366,)
        daily_pred_366_np = daily_pred_366[0, -1, :].cpu().detach().numpy()  # Last timestep only

        # Denormalize
        if sy is not None:
            daily_pred_366_np = sy.inverse_transform(daily_pred_366_np.reshape(1, -1)).flatten()

        # Get correct number of days for the specific year
        days_in_year = get_days_in_year(year)
        daily_pred = daily_pred_366_np[:days_in_year]

        return daily_pred

    def _predict_hourly(self, daily_sum: float, year: int, day_idx: int, historical_days=None):
        """
        Predict 24 hourly values for a given day with sequence context.

        CRITICAL: The current day to predict should be the LAST in the sequence!
        """
        device = self.ds.tc.device

        start_date = pd.Timestamp(f'{year}-01-01')
        current_date = start_date + pd.Timedelta(days=day_idx)

        # Build sequence with current day as the LAST element
        if historical_days is not None and len(historical_days) > 0:
            # Use T_seq_daily-1 historical days + current day
            # This ensures the current day is always at the end
            days_to_use = historical_days[-(self.ds.tc.T_seq_daily - 1):]
            days_to_use.append({'date': current_date, 'daily_sum': daily_sum})
        else:
            # No history - use current day only, padded if necessary
            if self.ds.tc.T_seq_daily == 1:
                days_to_use = [{'date': current_date, 'daily_sum': daily_sum}]
            else:
                # Pad with synthetic history
                days_to_use = []
                for i in range(self.ds.tc.T_seq_daily - 1, 0, -1):
                    synthetic_date = current_date - pd.Timedelta(days=i)
                    days_to_use.append({
                        'date': synthetic_date,
                        'daily_sum': daily_sum * 0.95  # Slight variation
                    })
                days_to_use.append({'date': current_date, 'daily_sum': daily_sum})

        # Ensure exactly T_seq_daily timesteps
        while len(days_to_use) < self.ds.tc.T_seq_daily:
            # Pad at the beginning with the first day
            first_day = days_to_use[0].copy()
            first_day['date'] = first_day['date'] - pd.Timedelta(days=1)
            days_to_use.insert(0, first_day)

        # Trim if too long (shouldn't happen but just in case)
        days_to_use = days_to_use[-self.ds.tc.T_seq_daily:]

        # Build dataframe with sequence
        df_seq = []
        for d_data in days_to_use:
            df_seq.append({
                'ds': d_data['date'],
                'date': d_data['date'].date() if hasattr(d_data['date'], 'date') else d_data['date'],
                'x_0': d_data['daily_sum']
            })

        df_dummy = pd.DataFrame(df_seq)

        # Add Fourier features - STANDARD ORDER: yearly, monthly, weekly, daily
        start_idx = 1

        # 1. Yearly
        if self.ds.fc.K_yearly_to_hourly > 0:
            df_dummy, _ = append_fourier_features_matrix(
                df_dummy,
                K=self.ds.fc.K_yearly_to_hourly,
                period_days=self.ds.fc.P_year,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += 2 * self.ds.fc.K_yearly_to_hourly * 24

        # 2. Monthly
        if self.ds.fc.K_monthly_to_hourly > 0:
            df_dummy, _ = append_fourier_features_matrix(
                df_dummy,
                K=self.ds.fc.K_monthly_to_hourly,
                period_days=self.ds.fc.P_month,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += 2 * self.ds.fc.K_monthly_to_hourly * 24

        # 3. Weekly
        if self.ds.fc.K_weekly_to_hourly > 0:
            df_dummy, _ = append_fourier_features_matrix(
                df_dummy,
                K=self.ds.fc.K_weekly_to_hourly,
                period_days=self.ds.fc.P_week,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += 2 * self.ds.fc.K_weekly_to_hourly * 24

        # 4. Daily
        if self.ds.fc.K_daily_to_hourly > 0:
            df_dummy, _ = append_fourier_features_matrix(
                df_dummy,
                K=self.ds.fc.K_daily_to_hourly,
                period_days=self.ds.fc.P_day,
                H=24,
                mode="matrix",
                start_idx=start_idx
            )
            start_idx += 2 * self.ds.fc.K_daily_to_hourly * 24

        # Extract features
        x_cols = ["x_0"] + sorted([c for c in df_dummy.columns if c.startswith("x_") and c != "x_0"],
                                  key=lambda s: int(s.split("_")[1]))
        X = df_dummy[x_cols].to_numpy(dtype=np.float32)

        # Normalize
        sx = self.ds.scalers['daily']['sx']
        sy = self.ds.scalers['daily']['sy']

        if sx is not None:
            X_norm = sx.transform(X).astype(np.float32)
        else:
            X_norm = X

        # Predict 24 hours
        X_tensor = torch.from_numpy(X_norm).unsqueeze(0).to(device)  # (1, T_seq_daily, F)
        hourly_pred, _ = self.ds.rnn_hourly(X_tensor)  # (1, T_seq_daily, 24)

        # CORRECT EXTRACTION: Take the LAST timestep (current day's hourly prediction)
        # Shape: (1, T_seq_daily, 24) -> (24,)
        hourly_pred_np = hourly_pred[0, -1, :].cpu().detach().numpy()  # Last timestep only!

        # Denormalize
        if sy is not None:
            hourly_pred_np = sy.inverse_transform(hourly_pred_np.reshape(1, -1)).flatten()

        return hourly_pred_np


# ============================================================================
# Uncertainty Quantification - Sampling
# ============================================================================

@dataclass
class SamplingConfig:
    """Configuration for hierarchical uncertainty sampling."""
    n_stage1: int = 5000  # Number of samples from Stage 1
    n_stage2: int = 10  # Number of samples per Stage 1 sample for Stage 2
    random_seed: int = 42


class HierarchicalSampler:
    """
    Hierarchical uncertainty sampler for 2-stage downscaling.

    Samples from Stage 1, then propagates through Stage 2 deterministically,
    then samples Stage 2 uncertainty for each day.
    """

    def __init__(
            self,
            downscaler: HierarchicalDownscaler_2Stage,
            sigma_stage1: np.ndarray,
            sigma_stage2: np.ndarray,
            sampling_config: SamplingConfig = None
    ):
        self.ds = get_base_model(downscaler)
        self.ds_wrapped = downscaler
        self.sigma1 = sigma_stage1  # (366, 366)
        self.sigma2 = sigma_stage2  # (24, 24)
        self.sc = sampling_config if sampling_config is not None else SamplingConfig()

        np.random.seed(self.sc.random_seed)

    def sample(self, yearly_sum: float, year: int, historical_years=None,
               n_stage1: int = None, n_stage2: int = None):
        """
        Generate hierarchical samples for a year.

        Args:
            yearly_sum: Yearly aggregate value
            year: Year to predict
            historical_years: Historical years context for Stage 1
            n_stage1: Number of Stage 1 samples (default: from config)
            n_stage2: Number of Stage 2 samples per Stage 1 sample (default: from config)

        Returns:
            Dictionary with samples and shapes
        """
        if n_stage1 is None:
            n_stage1 = self.sc.n_stage1
        if n_stage2 is None:
            n_stage2 = self.sc.n_stage2

        leap_year = is_leap_year(year)
        n_days = 366 if leap_year else 365

        print(f"Sampling for year {year} (leap={leap_year}, days={n_days})")
        print(f"Stage 1 samples: {n_stage1}, Stage 2 samples per Stage 1: {n_stage2}")
        print(f"Total samples: {n_stage1 * n_stage2}")

        # Step 1: Sample Stage 1 (Daily values)
        daily_samples = self._sample_stage1(yearly_sum, year, historical_years, n_stage1, n_days)
        # Shape: (n_stage1, n_days)

        print ('finish stage 1 sampling')

        # Step 2: For each Stage 1 sample, get Stage 2 deterministic + sample
        hourly_samples = []

        for i in range(n_stage1):
            if (i + 1) % 500 == 0:
                print(f"  Processing Stage 1 sample {i + 1}/{n_stage1}...")

            # Get deterministic hourly prediction for this daily sample
            hourly_det = self._stage2_deterministic(daily_samples[i], year)
            # Shape: (n_days, 24)
          
            # Sample Stage 2 noise n_stage2 times for all days
            hourly_sampled = self._sample_stage2(hourly_det, n_stage2, n_days)
            # Shape: (n_stage2, n_days, 24)

            hourly_samples.append(hourly_sampled)

        # Stack: (n_stage1, n_stage2, n_days, 24)
        hourly_samples = np.array(hourly_samples)

        # Reshape to (n_stage1 * n_stage2, n_days * 24)
        all_samples = hourly_samples.reshape(n_stage1 * n_stage2, n_days * 24)

        return {
            'samples': all_samples,
            'shape': (n_stage1 * n_stage2, n_days, 24),
            'year': year,
            'n_stage1': n_stage1,
            'n_stage2': n_stage2,
            'daily_samples': daily_samples  # For diagnostics
        }

    def _sample_stage1(self, yearly_sum: float, year: int, historical_years,
                       n_samples: int, n_days: int) -> np.ndarray:
        """
        Sample daily values from Stage 1.

        Returns:
            daily_samples: (n_samples, n_days) in original scale
        """
        # Get deterministic prediction
        predictor = HierarchicalPredictor_2Stage(self.ds_wrapped)
        predictor.reset_hidden_states()
        daily_det = predictor._predict_daily(yearly_sum, year, historical_years)  # (n_days,)

        # Pad to 366 if needed
        if n_days == 365:
            daily_det_366 = np.pad(daily_det, (0, 1), mode='edge')
        else:
            daily_det_366 = daily_det

        # Sample noise in original space (sigma was estimated in original space)
        # Sample from multivariate normal
        epsilon1 = np.random.multivariate_normal(
            mean=np.zeros(366),
            cov=self.sigma1[:366, :366],
            size=n_samples
        )  # (n_samples, 366)

        # Add noise to deterministic
        daily_samples = daily_det_366[None, :] + epsilon1

        # Truncate to actual days
        daily_samples = daily_samples[:, :n_days]

        return daily_samples

    def _stage2_deterministic(self, daily_values: np.ndarray, year: int) -> np.ndarray:
        """
        Get deterministic Stage 2 prediction for given daily values.

        Args:
            daily_values: (n_days,) daily sums
            year: year

        Returns:
            hourly_det: (n_days, 24) deterministic hourly predictions
        """
        predictor = HierarchicalPredictor_2Stage(self.ds_wrapped)
        predictor.reset_hidden_states()

        n_days = len(daily_values)
        hourly_predictions = []
        historical_days = []

        for day_idx in range(n_days):
            hourly_pred = predictor._predict_hourly(
                daily_sum=daily_values[day_idx],
                year=year,
                day_idx=day_idx,
                historical_days=historical_days.copy() if len(historical_days) > 0 else None
            )
            hourly_predictions.append(hourly_pred)

            # Update historical
            start_date = pd.Timestamp(f'{year}-01-01')
            current_date = start_date + pd.Timedelta(days=day_idx)
            historical_days.append({
                'date': current_date,
                'daily_sum': daily_values[day_idx]
            })

        return np.array(hourly_predictions)  # (n_days, 24)

    def _sample_stage2(self, hourly_det: np.ndarray, n_samples: int, n_days: int) -> np.ndarray:
        """
        Sample Stage 2 noise for all days.

        Args:
            hourly_det: (n_days, 24) deterministic predictions
            n_samples: number of samples to generate
            n_days: number of days

        Returns:
            samples: (n_samples, n_days, 24)
        """
        samples = []

        for j in range(n_samples):
            # Sample epsilon for EACH day independently using SAME distribution
            epsilon2_all_days = np.random.multivariate_normal(
                mean=np.zeros(24),
                cov=self.sigma2,
                size=n_days
            )  # (n_days, 24)

            # Add to deterministic
            sample_j = hourly_det + epsilon2_all_days

            samples.append(sample_j)

        return np.stack(samples)  # (n_samples, n_days, 24)