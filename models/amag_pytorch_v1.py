#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch port of AMAG (attention-augmented magnitude) for SeisBench.

- Compatible with your ETHZ_loader / 03a_train_phasenet training pipeline.
- Outputs per-timestep magnitude regression: key "magnitude".
- Lightweight efficient channel attention (ECA) in encoder/decoder.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from seisbench.models.base import WaveformModel


# -----------------------------
# Attention blocks (ECA / SE)
# -----------------------------
class SE1D(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps: [B, C, T]."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ECA1D(nn.Module):
    """Efficient Channel Attention for 1D: fast, no FC bottleneck."""

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        # conv over channels (implemented via 1x1 conv over temporal dim==1 then squeeze)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        # x: [B, C, T]
        y = self.pool(x)  # [B, C, 1]
        y = y.transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = torch.sigmoid(y).transpose(1, 2)  # [B, C, 1]
        return x * y


# -----------------------------
# Core blocks
# -----------------------------
class ConvBlock1D(nn.Module):
    """Conv1d -> BN -> LeakyReLU; with residual option."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 5,
        stride: int = 1,
        groups: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.residual = residual and (in_ch == out_ch) and (stride == 1)

    def forward(self, x):
        y = self.act(self.bn(self.conv(x)))
        if self.residual:
            y = y + x
        return y


class DownBlock(nn.Module):
    """Two conv blocks + optional attention + downsample."""

    def __init__(self, in_ch: int, out_ch: int, k: int, use_eca: bool = True):
        super().__init__()
        self.block1 = ConvBlock1D(in_ch, out_ch, k)
        self.block2 = ConvBlock1D(out_ch, out_ch, k)
        self.attn = ECA1D(out_ch) if use_eca else SE1D(out_ch)
        self.down = nn.AvgPool1d(kernel_size=2, ceil_mode=True)  # safe temporal halving

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.block1(x)
        y = self.block2(y)
        y = self.attn(y)
        y_down = self.down(y)
        return y, y_down  # skip, next


class UpBlock(nn.Module):
    """Upsample (interp) + conv + skip fuse + attention + conv."""

    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, k: int, use_eca: bool = True
    ):
        super().__init__()
        self.upconv = ConvBlock1D(
            in_ch, out_ch, k=1, stride=1
        )  # channel reduce before fuse
        self.fuse = ConvBlock1D(out_ch + skip_ch, out_ch, k)
        self.block = ConvBlock1D(out_ch, out_ch, k)
        self.attn = ECA1D(out_ch) if use_eca else SE1D(out_ch)

    def forward(self, x, skip):
        # Upsample by 2 with linear interpolation (channel-preserving), then 1x1 conv
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.upconv(x)

        # Crop/align if off-by-one due to pooling/ceil. Match skip length.
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                pad_l = diff // 2
                pad_r = diff - pad_l
                x = F.pad(x, (pad_l, pad_r))
            elif diff < 0:
                x = x[..., : skip.shape[-1]]

        y = torch.cat([skip, x], dim=1)
        y = self.fuse(y)
        y = self.attn(y)
        y = self.block(y)
        return y


# -----------------------------
# AMAG magnitude model
# -----------------------------
class AMAGMag(WaveformModel):
    """
    Attention-augmented magnitude estimator (PyTorch).

    Input:  [B, C, T] (e.g., C=3 for Z/N/E; normalized per SeisBench pipeline)
    Output: {"magnitude": [B, T]}  # per-timestep regression

    Notes:
    - Depth and filters mirror a light 1D U-Net.
    - Attention via ECA; switch to SE by setting use_eca=False.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 32,
        depth: int = 4,
        kernel_size: int = 5,
        use_eca: bool = True,
        norm: Optional[str] = "std",
        default_args: Optional[dict] = None,
    ):
        super().__init__(norm=norm, default_args=default_args)

        self.in_channels = in_channels
        self.depth = depth
        self.kernel_size = kernel_size

        self.stem = ConvBlock1D(
            in_channels, base_filters, k=kernel_size, stride=1, residual=False
        )

        # Encoder
        enc = []
        ch = base_filters
        for d in range(depth):
            enc.append(DownBlock(ch, ch * 2, k=kernel_size, use_eca=use_eca))
            ch *= 2
        self.encoder = nn.ModuleList(enc)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock1D(ch, ch, k=kernel_size),
            ConvBlock1D(ch, ch, k=kernel_size),
            ECA1D(ch) if use_eca else SE1D(ch),
        )

        # Decoder
        dec = []
        for d in reversed(range(depth)):
            skip_ch = base_filters * (2**d)
            dec.append(
                UpBlock(
                    in_ch=ch,
                    skip_ch=skip_ch,
                    out_ch=skip_ch,
                    k=kernel_size,
                    use_eca=use_eca,
                )
            )
            ch = skip_ch
        self.decoder = nn.ModuleList(dec)

        # Magnitude head -> [B, 1, T] -> squeeze to [B, T] in forward
        self.mag_head = nn.Conv1d(ch, 1, kernel_size=3, padding=1)

        # (Optional) pick head (disabled by default)
        # self.pick_head = nn.Conv1d(ch, 3, kernel_size=3, padding=1)  # e.g., PSN logits

        # Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, C, T] waveform batch (already normalized by SeisBench transforms).
        Returns:
            {"magnitude": [B, T]}
        """
        y = self.stem(x)

        skips = []
        z = y
        for block in self.encoder:
            skip, z = block(z)
            skips.append(skip)

        z = self.bottleneck(z)

        for block, skip in zip(self.decoder, reversed(skips)):
            z = block(z, skip)

        mag = self.mag_head(z).squeeze(1)  # [B, T]
        return {"magnitude": mag}


# -----------------------------
# Masked losses (onset-aware)
# -----------------------------
def masked_mse_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_threshold: float = 0.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combine MAE and MSE with onset-aware masking:
    - mask1: target > threshold (post-onset region)
    - mask0: complement (pre-onset) with reduced weight

    Args:
        pred, target: [B, T]
        mask_threshold: 0.0 for labels that are 0 pre-onset, >0 post-onset
        alpha: balance between MSE (alpha) and MAE (1-alpha)
    """
    assert pred.shape == target.shape
    mask1 = (target > mask_threshold).float()
    mask0 = 1.0 - mask1

    # weights: full post-onset, lightly penalize pre-onset drift
    w1 = 1.0
    w0 = 0.2

    mse = (pred - target) ** 2
    mae = (pred - target).abs()

    loss = alpha * (w1 * (mse * mask1).mean() + w0 * (mse * mask0).mean()) + (
        1 - alpha
    ) * (w1 * (mae * mask1).mean() + w0 * (mae * mask0).mean())
    return loss


# -----------------------------
# Small training adapter (optional)
# -----------------------------
def make_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(batch, model: AMAGMag, optimizer, device: torch.device):
    """
    Expects SeisBench-style batch dict:
        batch["X"]: [B, C, T] waveforms
        batch["magnitude"]: [B, T] labels (zero pre-onset, const mag post-onset)
    """
    x = batch["X"].to(device)
    y = batch["magnitude"].to(device)

    model.train()
    optimizer.zero_grad()
    out = model(x)["magnitude"]
    loss = masked_mse_mae(out, y, mask_threshold=0.0, alpha=0.5)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def eval_step(batch, model: AMAGMag, device: torch.device):
    x = batch["X"].to(device)
    y = batch["magnitude"].to(device)
    model.eval()
    out = model(x)["magnitude"]
    loss = masked_mse_mae(out, y, mask_threshold=0.0, alpha=0.5)
    return float(loss.detach().cpu())
