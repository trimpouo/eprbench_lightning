import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSpectralAttentionFeatureRefinement(nn.Module):
    """Cross-spectral Attention Feature Refinement (CAFR) module.

    This module performs adaptive cross-layer feature fusion by modelling
    channel-wise correlations between high-level semantic features and
    low-level detail features via a cross-spectral attention map.

    Reference:
        MODA: The First Challenging Benchmark for Multispectral Object
        Detection in Aerial Images (AAAI 2026).

    :param in_channels: Number of channels of input feature maps.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.low_level_feature_alignment = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=True
        )
        self.global_spectral_projection = nn.Linear(in_channels, in_channels, bias=True)

    def forward(
        self, high_level_feature: torch.Tensor, low_level_feature: torch.Tensor
    ) -> torch.Tensor:
        """Refine features via cross-spectral attention.

        :param high_level_feature: High-level feature map, shape (B, C, H, W).
        :param low_level_feature: Low-level feature map, shape (B, C, H', W').
            It will be resized to (H, W) internally.
        :return: Refined feature map, shape (B, C, H, W).
        """
        batch_size, channels, high_height, high_width = high_level_feature.shape

        aligned_low = self.low_level_feature_alignment(low_level_feature)
        aligned_low = F.interpolate(
            aligned_low,
            size=(high_height, high_width),
            mode="bilinear",
            align_corners=False,
        )

        high_descriptor = F.adaptive_avg_pool2d(
            high_level_feature, output_size=1
        ).flatten(1)
        low_descriptor = F.adaptive_avg_pool2d(aligned_low, output_size=1).flatten(1)

        high_spectral = self.global_spectral_projection(high_descriptor).unsqueeze(-1)
        low_spectral = self.global_spectral_projection(low_descriptor).unsqueeze(-1)

        cross_attn = torch.matmul(high_spectral, low_spectral.transpose(1, 2))
        cross_attn = F.softmax(cross_attn, dim=-1)

        high_weight = torch.matmul(
            cross_attn.transpose(-1, -2), high_spectral
        ).unsqueeze(-1)
        low_weight = torch.matmul(cross_attn, low_spectral).unsqueeze(-1)

        refined = high_weight * high_level_feature + low_weight * aligned_low
        return refined
