
# amag_torch.py
# Pure PyTorch reimplementation of the Keras AMAG (attention-based magnitude) model
# - Model architecture (Conv2D -> Downsample) x D -> LSTM -> Self-Attn -> (Conv + Upsample)xD -> Conv
# - Dataset/Dataloader matching the original generators (HDF5 + CSV, taper + optional bandpass)
# - Train / Evaluate / Predict entrypoints
# Author: ChatGPT (adapted for Chris)
# Date: 2025-09-25

from __future__ import annotations
import argparse
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional SciPy (for bandpass). If missing, we silently skip filtering.
try:
    from scipy.signal import butter, filtfilt
    _SCIPY = True
except Exception:
    _SCIPY = False
    warnings.warn("SciPy not available; bandpass filtering will be skipped.", RuntimeWarning)

# ------------------------
# Utils
# ------------------------

def _leaky_relu():
    return nn.LeakyReLU(0.1, inplace=True)

def _same_padding_1d(k: int) -> int:
    """Return 'same' padding for a 1D kernel along time dimension."""
    return k // 2

def crop_time_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Center-crop (or left-crop) x along time (dim=2) to match ref's time length.
    (Keras code slices without concat; we mirror that behavior.)
    Shapes: x:[B,C,T,1], ref:[B,C',T_ref,1]
    """
    T = x.shape[2]
    T_ref = ref.shape[2]
    if T == T_ref:
        return x
    if T < T_ref:
        # pad (rare) to the left to match
        pad = (0, 0, 0, 0, 0, T_ref - T)  # (W_left,W_right, H_top,H_bottom, T_left,T_right) but for 3D it's slightly different.
        # We can't use F.pad for NCT1; use temporal padding on dim=2
        pad_amount = T_ref - T
        x = F.pad(x, (0, 0, 0, 0, 0, pad_amount))  # pad time on the right
        return x
    # If longer, slice the left-most T_ref samples (Keras tf.slice default offsets=0)
    return x[:, :, :T_ref, :]

# ------------------------
# Attention block (simple self-attention)
# ------------------------

class SelfAttention1D(nn.Module):
    """
    Simple scaled dot-product self-attention with 1 head (batch_first).
    Input: [B, T, C] -> Output: [B, T, C]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x)  # [B,T,C]
        k = self.k(x)
        v = self.v(x)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(C)  # [B,T,T]
        attn = torch.softmax(attn_scores, dim=-1)  # closest to keras-self-attention default
        out = torch.matmul(attn, v)  # [B,T,C]
        out = self.proj(out)
        return out

# ------------------------
# AMAG model
# ------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride_t: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(k, 1),
                              stride=(stride_t, 1),
                              padding=(_same_padding_1d(k), 0), bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = _leaky_relu()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ConvUpBNAct(nn.Module):
    """Conv (stride=1) then Nearest-Neighbor upsample by factor 2 in time dim."""
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(k, 1),
                              stride=(1, 1),
                              padding=(_same_padding_1d(k), 0), bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = _leaky_relu()
    def forward(self, x):
        x = self.conv(x)
        # Upsample only along time dimension by 2 (match Keras UpSampling2D(size=(2,1)))
        x = F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        return self.act(self.bn(x))

class AMAGTorch(nn.Module):
    """
    PyTorch version of AMAG.
    Input:  X [B, T, 1, 3]  (channels-last like Keras; we permute internally)
    Output: Y [B, T, 1, 1]  (time-distributed magnitude)
    """
    def __init__(self, in_channels: int = 3, base_filters: int = 8, k: int = 7, depths: int = 5, out_channels: int = 1):
        super().__init__()
        self.depths = depths
        # Encoder
        enc_blocks = []
        ch_in = in_channels
        for d in range(depths):
            ch_out = base_filters * (2 ** d)
            # first conv at each depth (stride 1)
            enc_blocks.append(ConvBNAct(ch_in, ch_out, k, stride_t=1))
            ch_in = ch_out
            # save a second conv per depth before downsample (to match Keras's Conv2d_BN1 then Conv2d_BN2 on same filters)
            if d < depths - 1:
                # downsample in time by stride=2
                enc_blocks.append(ConvBNAct(ch_in, ch_out, k, stride_t=2))
        self.encoder = nn.ModuleList(enc_blocks)

        # LSTM + Self-Attention
        self.lstm_hidden = ch_in
        self.lstm = nn.LSTM(input_size=ch_in, hidden_size=ch_in, num_layers=1, batch_first=True, bidirectional=False)
        self.attn = SelfAttention1D(ch_in)

        # Decoder (mirror)
        dec_blocks = []
        for d in reversed(range(depths - 1)):  # depths-2 down to 0 (same as Keras loop)
            ch_out = base_filters * (2 ** d)
            dec_blocks.append(ConvUpBNAct(ch_in, ch_out, k))
            ch_in = ch_out
        self.decoder = nn.ModuleList(dec_blocks)

        # Final 3x1 conv head
        self.head = nn.Conv2d(ch_in, out_channels, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 1, C]  (match the Keras tensors fed by the original generator)
        """
        # to NCHW: [B,C,T,1]
        x = x.permute(0, 3, 1, 2).contiguous()

        # Encode, keep feature after each stride-1 conv (the Keras code keeps convs per depth)
        feats = []
        i = 0
        for d in range(self.depths):
            # stride-1 conv
            x = self.encoder[i](x); i += 1
            feats.append(x)
            # optional downsample (not at last depth)
            if d < self.depths - 1:
                x = self.encoder[i](x); i += 1  # stride-2 conv

        # Reshape to sequence: [B, C, T, 1] -> [B, T, C]
        B, C, T, _ = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B, T, C)

        # LSTM + Self-Attn
        seq, _ = self.lstm(seq)          # [B,T,C]
        seq = self.attn(seq)             # [B,T,C]

        # Back to [B,C,T,1]
        x = seq.reshape(B, T, C).permute(0, 2, 1).unsqueeze(-1)  # [B,C,T,1]

        # Decode (Conv + Upsample) and slice to "skip" spatial size. NOTE: Keras code crops to skip but doesn't concat.
        for d, dec in enumerate(self.decoder):
            x = dec(x)
            ref = feats[self.depths - 2 - d]  # mirror depth index
            x = crop_time_like(x, ref)

        y = self.head(x)  # [B,1,T,1]
        # Return channels-last like Keras: [B,T,1,1]
        y = y.permute(0, 2, 3, 1).contiguous()
        return y

# ------------------------
# Dataset
# ------------------------

def _taper(sig: np.ndarray, N: int = 100) -> np.ndarray:
    """Symmetric cosine taper ('Hann-like')."""
    n = sig.shape[-1]
    w = math.pi / N
    F0, F1 = 0.5, 0.5
    win = np.ones((n,), dtype=np.float32)
    win[:N] = (F0 - F1 * np.cos(w * (np.arange(N)))).astype(np.float32)
    win2 = win[::-1].copy()
    sig = sig * win
    sig = sig * win2
    return sig

def _bandpass(data: np.ndarray, n: int = 2, f1: float = 1.0, f2: float = 20.0, dt: float = 0.01) -> np.ndarray:
    """Butterworth bandpass, if SciPy is available."""
    if not _SCIPY:
        return data
    nyq = 0.5 / dt
    b, a = butter(n, [f1 / nyq, f2 / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

@dataclass
class MagDatasetConfig:
    h5_path: str
    csv_path: str
    ev_list: List[str]
    dpss: int = 300
    noi_win: int = 300
    lm: float = 0.0         # label fill value pre-P
    add_one: bool = True    # label = mag + 1.0 (matches Keras generators)
    random_window: bool = True  # emulate ratio mode (random pre-window before P)
    tpr_taper: bool = False

class MagDataset(Dataset):
    """
    Pairs (X, Y) where:
      - X: float32 [T, 1, 3]   (time, pseudo-height=1, channels=3 ENZ)
      - Y: float32 [T, 1, 1]   (time-distributed magnitude; baseline 'lm' pre-P, constant mag after P)
    """
    def __init__(self, cfg: MagDatasetConfig, df: Optional[pd.DataFrame] = None):
        super().__init__()
        self.cfg = cfg
        self.h5 = None  # opened lazily per __getitem__
        if df is None:
            self.df = pd.read_csv(cfg.csv_path)
        else:
            self.df = df
        self.tt = cfg.dpss + cfg.noi_win

    def __len__(self) -> int:
        return len(self.cfg.ev_list)

    def _open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.cfg.h5_path, 'r')

    def __getitem__(self, idx: int):
        self._open()
        name = self.cfg.ev_list[idx]
        # Read waveform: shape [3, N] or [N, 3]
        arr = np.array(self.h5[f"data/{name}"])
        if arr.shape[0] != 3 and arr.shape[1] == 3:
            arr = arr.T  # [3,N]
        # Taper each channel
        for i in range(3):
            arr[i, :] = _taper(arr[i, :], 100)
        # Bandpass 1-20 Hz, dt=0.01
        arr = _bandpass(arr, n=2, f1=1.0, f2=20.0, dt=0.01)  # [3,N]

        # Get metadata: P index, magnitude
        row = self.df.loc[self.df['trace_name'] == name].iloc[0]
        p_idx = int(row['trace_P_arrival_sample'] if 'trace_P_arrival_sample' in row else row.get('p_arrival_sample'))
        mag = float(row['source_magnitude'])
        if self.cfg.add_one:
            mag = mag + 1.0

        # Choose start so that window length = noi_win + dpss, ending after P by random amount
        if self.cfg.random_window:
            # emulate Keras: ran_win is 100..(tt-100) step 1
            ran_min, ran_max = 100, (self.tt // 100 - 1) * 100
            ran_win = int(np.random.randint(ran_min, max(ran_min+1, ran_max)))
            start = max(0, p_idx - ran_win)
            rel_p = p_idx - start
        else:
            start = max(0, p_idx - self.cfg.dpss - self.cfg.noi_win)
            rel_p = p_idx - start

        if start + self.tt > arr.shape[1]:
            # If near the end, pad with zeros
            pad = start + self.tt - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)), mode='constant')

        clip = arr[:, start:start + self.tt]  # [3, T]
        X = np.empty((self.tt, 1, 3), dtype=np.float32)
        X[:, 0, :] = clip.T

        # Labels: baseline lm until P (or ran_win), then constant mag
        Y = np.full((self.tt, 1, 1), self.cfg.lm, dtype=np.float32)
        tail_start = max(0, int(rel_p))
        Y[tail_start:, 0, 0] = mag

        return torch.from_numpy(X), torch.from_numpy(Y)

# ------------------------
# Losses
# ------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)

def wmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Value-aware weighting similar to the Keras code:
    - 0<y<=0.5: x2
    - 2<y<=3: x4
    - 3<y<=4: x8
    - 4<y<=5: x16
    - 5<y: x32
    """
    y = target
    w = torch.ones_like(y)
    w = torch.where((y > 0.0) & (y <= 0.5), w * 2.0, w)
    w = torch.where((y > 2.0) & (y <= 3.0), w * 4.0, w)
    w = torch.where((y > 3.0) & (y <= 4.0), w * 8.0, w)
    w = torch.where((y > 4.0) & (y <= 5.0), w * 16.0, w)
    w = torch.where((y > 5.0), w * 32.0, w)
    return ((w * (pred - target) ** 2).mean())

def mse_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target) + F.l1_loss(pred, target)

def get_loss(name: str):
    name = name.lower()
    if name == "mse":
        return mse_loss
    if name == "wmse":
        return wmse_loss
    if name in ("mse_mae", "mse+mae"):
        return mse_mae_loss
    raise ValueError(f"Unknown loss: {name}")

# ------------------------
# Training & Evaluation
# ------------------------

def train_one_epoch(model, dl, opt, loss_fn, device):
    model.train()
    total = 0.0
    for X, Y in dl:
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * X.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, loss_fn, device):
    model.eval()
    total = 0.0
    for X, Y in dl:
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        loss = loss_fn(pred, Y)
        total += loss.item() * X.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def predict_all(model, dl, device):
    model.eval()
    preds = []
    for X, _ in dl:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0)

# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 with /data/<trace_name> arrays")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with metadata (trace_name, source_magnitude, P index)")
    parser.add_argument("--save_dir", type=str, default="./out", help="Where to save model & outputs")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--base_filters", type=int, default=8)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "wmse", "mse_mae"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dpss", type=int, default=300, help="len_signal")
    parser.add_argument("--noi_win", type=int, default=300, help="len_noise")
    parser.add_argument("--add_one", action="store_true", help="Label = magnitude + 1.0 (recommended to mirror Keras)")
    parser.add_argument("--no_add_one", dest="add_one", action="store_false")
    parser.set_defaults(add_one=True)
    parser.add_argument("--random_window", action="store_true", help="Random pre-P window (like ratio mode).")
    parser.add_argument("--no_random_window", dest="random_window", action="store_false")
    parser.set_defaults(random_window=True)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Read CSV & build event list like the original
    df = pd.read_csv(args.csv_path)
    # If STEAD-like schema, clean SNR and filter events
    snr_col = None
    if "snr_db" in df.columns:
        # Keras script parses snr_db like "['x y z']" -> float
        try:
            df['snr_db'] = df['snr_db'].astype(str).str.split(r'[\[\]\s\n]+').str[-2].astype(float)
            snr_col = "snr_db"
        except Exception:
            pass
        mask = (df.get('source_magnitude_type', 'ml').str.lower() == 'ml') & (df['source_magnitude'] > 0) & (df.get(snr_col, 100) > 10)
    else:
        # INSTANCE-like
        mask = (df.get('source_magnitude_type', 'ML').str.upper() == 'ML') & (df['source_magnitude'] > 0) & (df.get('trace_Z_snr_db', 100) > 10)

    ev_list = df.loc[mask, 'trace_name'].astype(str).tolist()
    rng = np.random.default_rng(7)
    rng.shuffle(ev_list)

    n = len(ev_list)
    n_train = int(0.8 * n)
    n_valid = int(0.9 * n)
    train_list = ev_list[:n_train]
    valid_list = ev_list[n_train:n_valid]
    test_list  = ev_list[n_valid:]

    cfg_train = MagDatasetConfig(
        h5_path=args.data_path, csv_path=args.csv_path, ev_list=train_list,
        dpss=args.dpss, noi_win=args.noi_win, add_one=args.add_one, random_window=args.random_window
    )
    cfg_valid = MagDatasetConfig(
        h5_path=args.data_path, csv_path=args.csv_path, ev_list=valid_list,
        dpss=args.dpss, noi_win=args.noi_win, add_one=args.add_one, random_window=args.random_window
    )
    cfg_test = MagDatasetConfig(
        h5_path=args.data_path, csv_path=args.csv_path, ev_list=test_list,
        dpss=args.dpss, noi_win=args.noi_win, add_one=args.add_one, random_window=args.random_window
    )

    ds_train = MagDataset(cfg_train, df=df)
    ds_valid = MagDataset(cfg_valid, df=df)
    ds_test  = MagDataset(cfg_test,  df=df)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMAGTorch(in_channels=3, base_filters=args.base_filters, k=args.kernel_size, depths=args.depth, out_channels=1).to(device)
    loss_fn = get_loss(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float('inf')
    best_path = os.path.join(args.save_dir, "amag_torch.pt")

    if not args.test_only:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, dl_train, opt, loss_fn, device)
            val_loss = evaluate(model, dl_valid, loss_fn, device)
            print(f"Epoch {epoch:03d}: train {train_loss:.5f} | valid {val_loss:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(),
                            "cfg": vars(args)}, best_path)
                print(f"  -> Saved best model to {best_path}")

    # Inference (on best model if available)
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        print(f"Loaded best checkpoint (val loss={best_val:.5f}) from {best_path}")
    else:
        print("No checkpoint found; using current model weights.")

    preds = predict_all(model, dl_test, device=device)  # [N, T, 1, 1]
    np.save(os.path.join(args.save_dir, "preds_test.npy"), preds)

    # Derive event-level magnitude by taking the max across time (matches the visual evaluation used in Keras notebook)
    preds_event = preds[..., 0].max(axis=1)  # [N]
    np.save(os.path.join(args.save_dir, "preds_event_max.npy"), preds_event)

    print("Done. Files saved in:", args.save_dir)


if __name__ == "__main__":
    main()
