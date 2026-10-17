"""Temporal Event Frame Aggregator (TEFA) module.

Aggregates multi-frame event representations across the temporal dimension
using lightweight temporal attention with a mean-pooling residual baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEventFrameAggregator(nn.Module):
    """Aggregate temporal sequences of event frame features or descriptors.

    Supports both sequence descriptors ``[B, T, C]`` and spatial feature maps
    ``[B, T, C, H, W]``. Starts as exact mean pooling via a zero-initialized
    residual scale.

    :param channels: Feature dimension ``C``.
    :param reduction: Hidden channel reduction ratio for temporal scoring.
    """

    def __init__(self, channels: int = 768, reduction: int = 8) -> None:
        super().__init__()
        self.channels = channels
        hidden = max(16, channels // reduction)
        self.score_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate across the temporal dimension ``T`` (dim=1).

        :param x: Input tensor of shape ``[B, T, C]`` or ``[B, T, C, H, W]``.
        :return: Aggregated tensor of shape ``[B, C]`` or ``[B, C, H, W]``.
        """
        if x.dim() == 5:
            pooled = x.mean(dim=(-2, -1))
            logits = self.score_net(pooled)
            weights = F.softmax(logits, dim=1).unsqueeze(-1).unsqueeze(-1)
            attn_out = (x * weights).sum(dim=1)
            mean_out = x.mean(dim=1)
            return mean_out + self.res_scale * (attn_out - mean_out)

        if x.dim() == 3:
            logits = self.score_net(x)
            weights = F.softmax(logits, dim=1)
            attn_out = (x * weights).sum(dim=1)
            mean_out = x.mean(dim=1)
            return mean_out + self.res_scale * (attn_out - mean_out)

        raise ValueError(f"Expected 3D or 5D tensor for temporal aggregation, got {x.shape}")
