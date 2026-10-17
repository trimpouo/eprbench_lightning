"""Information-Gated Reconstruction (IGR) module.

Separates feature maps into high-information and low-information components
via learnable group-wise normalization and sigmoid-gated masking, then
reconstructs an enhanced feature map by selectively boosting informative
regions and suppressing noise.
"""

import torch
import torch.nn as nn


class GroupBatchNorm2d(nn.Module):
    """Group-wise batch normalization that computes statistics per channel group.

    :param num_channels: Total number of input channels.
    :param num_groups: Number of groups to split channels into.
    :param eps: Small constant for numerical stability.
    """

    def __init__(self, num_channels: int, num_groups: int = 16, eps: float = 1e-10):
        super().__init__()
        assert num_channels >= num_groups, (
            "num_channels must be >= num_groups"
        )
        self.num_groups = num_groups
        self.scale = nn.Parameter(torch.randn(num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group-wise normalization.

        :param x: Input tensor of shape (B, C, H, W).
        :return: Normalized tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        grouped_feat = x.reshape(B, self.num_groups, -1)

        group_mean = grouped_feat.mean(dim=2, keepdim=True)
        group_std = grouped_feat.std(dim=2, keepdim=True)

        normalized_feat = (grouped_feat - group_mean) / (group_std + self.eps)
        normalized_feat = normalized_feat.reshape(B, C, H, W)

        return normalized_feat * self.scale + self.shift


class InfoGatedReconstructUnit(nn.Module):
    """Information-Gated Reconstruction Unit.

    Separates input features into high-information and low-information
    components via a learnable sigmoid gate, then reconstructs an enhanced
    feature map by amplifying informative regions and attenuating noise.

    :param in_channels: Number of input channels.
    :param gate_threshold: Threshold for binary high/low information split.
    :param num_groups: Number of groups for GroupBatchNorm2d.
    """

    def __init__(
        self,
        in_channels: int,
        gate_threshold: float = 0.5,
        num_groups: int = 16,
    ):
        super().__init__()
        self.group_norm = GroupBatchNorm2d(in_channels, num_groups=num_groups)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct enhanced features via information gating.

        :param x: Input feature map of shape (B, C, H, W).
        :return: Enhanced feature map of shape (B, C, H, W).
        """
        norm_feat = self.group_norm(x)

        channel_weights = self.group_norm.scale / self.group_norm.scale.sum()
        info_scores = self.sigmoid(norm_feat * channel_weights)

        high_info_mask = (info_scores >= self.gate_threshold).to(x.dtype)
        low_info_mask = (info_scores < self.gate_threshold).to(x.dtype)

        high_info_feat = high_info_mask * x
        low_info_feat = low_info_mask * x

        # Reconstruct: boost high-info, suppress low-info, add residual
        return info_scores * high_info_feat + (1 - info_scores) * low_info_feat + x