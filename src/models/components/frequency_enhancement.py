import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(channels: int, max_groups: int = 32) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if g <= max_groups and channels % g == 0:
            return g
    return 1


def _rfft_masks(h: int, w_rfft: int, device, dtype, low_cut: float = 0.25, mid_cut: float = 0.60):
    """Build low/mid/high radial masks for rfft2 output.

    Returns shape [1, 1, 3, H, W_rfft].
    """
    w = (w_rfft - 1) * 2
    fy = torch.fft.fftfreq(h, d=1.0, device=device).abs().view(h, 1)
    fx = torch.fft.rfftfreq(w, d=1.0, device=device).abs().view(1, w_rfft)
    radius = torch.sqrt(fy * fy + fx * fx)
    radius = radius / radius.max().clamp_min(1e-6)

    low = (radius <= low_cut).to(dtype)
    mid = ((radius > low_cut) & (radius <= mid_cut)).to(dtype)
    high = (radius > mid_cut).to(dtype)
    return torch.stack([low, mid, high], dim=0).view(1, 1, 3, h, w_rfft)


class FFTBandReweight(nn.Module):
    """Native PyTorch FFT frequency-band reweighting.

    Input/Output: [B, C, H, W].
    Uses float32 FFT under AMP, then casts back to input dtype.
    """

    def __init__(self, dim: int, low_cut: float = 0.25, mid_cut: float = 0.60, max_gain: float = 0.50):
        super().__init__()
        self.low_cut = low_cut
        self.mid_cut = mid_cut
        self.max_gain = max_gain
        self.band_logits = nn.Parameter(torch.zeros(1, dim, 3, 1, 1))
        with torch.no_grad():
            # mild high-frequency prior for event edge/texture details
            self.band_logits[:, :, 2].fill_(0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        b, c, h, w = x.shape

        xf = torch.fft.rfft2(x.float(), norm="ortho")
        masks = _rfft_masks(h, xf.shape[-1], xf.device, xf.real.dtype, self.low_cut, self.mid_cut)

        gains = torch.tanh(self.band_logits.float())
        scale = 1.0 + self.max_gain * (gains * masks).sum(dim=2)

        yf = xf * scale
        y = torch.fft.irfft2(yf, s=(h, w), norm="ortho")
        return y.to(dtype)


class HighFrequencyGate(nn.Module):
    """High-frequency spatial + channel gate.

    Inspired by DCT high-frequency perception and multi-frequency channel attention.
    Input:  [B, C, H, W]
    Output: enhanced high-frequency delta [B, C, H, W], energy map [B, H, W]
    """

    def __init__(self, dim: int, highpass_cut: float = 0.25, reduction: int = 16):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.highpass_cut = highpass_cut

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(_valid_groups(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )

    def _highpass(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        masks = _rfft_masks(h, xf.shape[-1], xf.device, xf.real.dtype, low_cut=self.highpass_cut, mid_cut=0.60)
        high_mask = 1.0 - masks[:, :, 0]
        high = torch.fft.irfft2(xf * high_mask, s=(h, w), norm="ortho")
        return high.to(dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        high = self._highpass(x)
        high_abs = high.abs()

        channel_attn = torch.sigmoid(
            self.channel_mlp(F.adaptive_avg_pool2d(high_abs, 1))
            + self.channel_mlp(F.adaptive_max_pool2d(high_abs, 1))
        )

        spatial_in = torch.cat(
            [high.mean(dim=1, keepdim=True), high_abs.amax(dim=1, keepdim=True)],
            dim=1,
        )
        spatial_attn = torch.sigmoid(self.spatial(spatial_in))

        delta = self.proj(high * channel_attn * spatial_attn)
        energy = high_abs.mean(dim=1)  # [B, H, W]
        return delta, energy


class EventFrequencyEnhancer(nn.Module):
    """Recommended event feature frequency enhancement plug-in.

    It should be inserted after DINOv2 patch tokens are reshaped to [B, C, Hp, Wp].
    The final residual scale gamma_out is initialized to 0, so initial behavior is
    equivalent to the original model.
    """

    def __init__(
        self,
        dim: int = 768,
        low_cut: float = 0.25,
        mid_cut: float = 0.60,
        highpass_cut: float = 0.25,
        max_gain: float = 0.50,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(_valid_groups(dim), dim)
        self.band = FFTBandReweight(dim, low_cut=low_cut, mid_cut=mid_cut, max_gain=max_gain)
        self.high_gate = HighFrequencyGate(dim, highpass_cut=highpass_cut)
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gamma_out = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, patch_map: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.norm(patch_map)
        x = self.band(x)
        high_delta, energy_map = self.high_gate(x)
        delta = self.fuse(high_delta)

        out = patch_map + self.gamma_out.to(patch_map.dtype) * delta

        freq_energy = energy_map.flatten(1)
        freq_energy = (freq_energy - freq_energy.mean(dim=1, keepdim=True)) / (
            freq_energy.std(dim=1, keepdim=True) + 1e-6
        )

        return out, {
            "freq_energy": freq_energy,          # [B, N], for top-k scoring
            "freq_energy_map": energy_map,       # [B, Hp, Wp], for visualization/debug
            "freq_gamma": self.gamma_out.detach(),
        }


def tokens_to_map(patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Convert [B, N, C] tokens to [B, C, H, W]. Requires square patch grid."""
    b, n, c = patch_tokens.shape
    h = int(math.sqrt(n))
    if h * h != n:
        raise ValueError(
            f"Patch token count {n} is not a square. Remove cls/register tokens first or pass explicit grid size."
        )
    patch_map = patch_tokens.transpose(1, 2).contiguous().view(b, c, h, h)
    return patch_map, h, h


def map_to_tokens(patch_map: torch.Tensor) -> torch.Tensor:
    """Convert [B, C, H, W] feature map back to [B, H*W, C]."""
    return patch_map.flatten(2).transpose(1, 2).contiguous()


class FrequencyContrastiveLoss(nn.Module):
    """Place-label adaptation of frequency contrastive regularization.

    Anchor/positive: two samples from the same place_id.
    Negative: sample from a different place_id.
    """

    def __init__(self, num_negatives: int = 2, eps: float = 1e-7):
        super().__init__()
        self.num_negatives = num_negatives
        self.eps = eps

    @staticmethod
    def _fft_amp(x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        return torch.log1p(torch.abs(xf))

    def forward(self, feat_map: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        b = feat_map.shape[0]
        if b <= 2:
            return feat_map.new_zeros(())

        amp = self._fft_amp(feat_map)
        all_idx = torch.arange(b, device=labels.device)
        losses = []

        for i in range(b):
            pos_idx = all_idx[(labels == labels[i]) & (all_idx != i)]
            neg_idx = all_idx[labels != labels[i]]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            p = pos_idx[torch.randint(pos_idx.numel(), (1,), device=labels.device)].item()
            neg_perm = torch.randperm(neg_idx.numel(), device=labels.device)[: self.num_negatives]
            negs = neg_idx[neg_perm]

            d_ap = F.l1_loss(amp[i], amp[p], reduction="mean")
            d_an = torch.stack([F.l1_loss(amp[i], amp[j], reduction="mean") for j in negs]).mean()
            losses.append(d_ap / (d_an + self.eps))

        if not losses:
            return feat_map.new_zeros(())
        return torch.stack(losses).mean().to(feat_map.dtype)
