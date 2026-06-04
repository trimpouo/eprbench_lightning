"""Spectral-guided Adaptive Cross-layer Fusion (SACF) module.

Reference:
    MODA: The First Challenging Benchmark for Multispectral Object
    Detection in Aerial Images (AAAI 2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanSpectralFeatureAggregator(nn.Module):
    """Aggregate spectral features via local Euclidean-distance similarity.

    :param local_window_size: Size of the local neighbourhood window.
    """

    def __init__(self, local_window_size: int = 3):
        super().__init__()
        self.local_window_size = local_window_size
        self.local_patch_unfold = nn.Unfold(
            kernel_size=local_window_size,
            padding=local_window_size // 2,
        )

    def forward(
        self,
        center_spectral_feature: torch.Tensor,
        reference_spectral_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = center_spectral_feature.shape
        num_neighbors = self.local_window_size * self.local_window_size
        num_positions = height * width

        local_spectral_patches = self.local_patch_unfold(
            reference_spectral_feature
        ).reshape(batch_size, channels, num_neighbors, num_positions)

        center_spectral_vectors = center_spectral_feature.reshape(
            batch_size, channels, num_positions
        )

        local_spectral_vectors = local_spectral_patches.permute(0, 3, 2, 1)
        center_spectral_vectors_for_distance = center_spectral_vectors.permute(
            0, 2, 1
        ).unsqueeze(2)

        euclidean_distance = torch.norm(
            local_spectral_vectors - center_spectral_vectors_for_distance,
            dim=-1,
        )

        spectral_similarity_weight = F.softmax(-euclidean_distance, dim=-1)

        aggregated_spectral_vectors = torch.matmul(
            spectral_similarity_weight.unsqueeze(2),
            local_spectral_vectors,
        )

        aggregated_spectral_feature = (
            aggregated_spectral_vectors.squeeze(2)
            .permute(0, 2, 1)
            .reshape(batch_size, channels, height, width)
        )

        return aggregated_spectral_feature + center_spectral_feature


class SpatialDetailEnhancer(nn.Module):
    """Enhance spatial details via high-frequency attention.

    :param in_channels: Number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.low_frequency_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.low_frequency_refine_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.high_frequency_attention_conv = nn.Conv2d(
            in_channels, 1, kernel_size=1, bias=False
        )
        self.detail_fusion_conv = nn.Conv2d(
            in_channels * 2, in_channels, kernel_size=1, bias=False
        )

    def forward(self, spatial_feature: torch.Tensor) -> torch.Tensor:
        low_frequency_feature = self.low_frequency_pool(spatial_feature)
        refined_low_frequency_feature = self.low_frequency_refine_conv(
            low_frequency_feature
        )
        refined_low_frequency_feature = F.interpolate(
            refined_low_frequency_feature,
            size=spatial_feature.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        upsampled_low_frequency_feature = F.interpolate(
            self.low_frequency_pool(spatial_feature),
            size=spatial_feature.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        high_frequency_detail_feature = (
            spatial_feature - upsampled_low_frequency_feature
        )

        high_frequency_attention = torch.sigmoid(
            self.high_frequency_attention_conv(high_frequency_detail_feature)
        )

        enhanced_high_frequency_feature = (
            high_frequency_detail_feature * high_frequency_attention
            + high_frequency_detail_feature
        )

        fused_spatial_detail_feature = self.detail_fusion_conv(
            torch.cat(
                [enhanced_high_frequency_feature, refined_low_frequency_feature],
                dim=1,
            )
        )

        return fused_spatial_detail_feature + spatial_feature


class SpectralSpatialAdaptiveFusion(nn.Module):
    """Spectral-guided Adaptive Cross-layer Fusion module.

    Aggregates high-level spectral features and enhances low-level spatial
    details, then fuses them via learnable adaptive weights.

    :param in_channels: Number of input channels.
    :param local_window_size: Size of the local window for spectral aggregation.
    """

    def __init__(self, in_channels: int, local_window_size: int = 3):
        super().__init__()
        self.spectral_feature_aggregator = EuclideanSpectralFeatureAggregator(
            local_window_size=local_window_size
        )
        self.spatial_detail_enhancer = SpatialDetailEnhancer(
            in_channels=in_channels
        )
        self.adaptive_fusion_weight_generator = nn.Conv2d(
            2, 2, kernel_size=1, bias=False
        )
        self.output_refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        high_level_spectral_feature: torch.Tensor,
        low_level_spatial_feature: torch.Tensor,
    ) -> torch.Tensor:
        aggregated_high_level_spectral_feature = self.spectral_feature_aggregator(
            high_level_spectral_feature, high_level_spectral_feature
        )
        aligned_high_level_spectral_feature = F.interpolate(
            aggregated_high_level_spectral_feature,
            size=low_level_spatial_feature.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        enhanced_low_level_spatial_feature = self.spatial_detail_enhancer(
            low_level_spatial_feature
        )

        spectral_global_response_map = aligned_high_level_spectral_feature.mean(
            dim=1, keepdim=True
        )
        spatial_global_response_map = enhanced_low_level_spatial_feature.mean(
            dim=1, keepdim=True
        )

        adaptive_fusion_weight = torch.sigmoid(
            self.adaptive_fusion_weight_generator(
                torch.cat(
                    [spectral_global_response_map, spatial_global_response_map],
                    dim=1,
                )
            )
        )

        spectral_fusion_weight = adaptive_fusion_weight[:, 0:1]
        spatial_fusion_weight = adaptive_fusion_weight[:, 1:2]

        fused_spectral_spatial_feature = (
            spectral_fusion_weight * aligned_high_level_spectral_feature
            + spatial_fusion_weight * enhanced_low_level_spatial_feature
        )

        output_feature = self.output_refine_conv(fused_spectral_spatial_feature)
        return output_feature


class EventStabilitySACF(nn.Module):
    """Event-Stability Guided SACF (ES-SACF).

    Extends the original SACF with event-cue-guided fusion and a residual
    scale initialized to zero. The fusion gate uses softmax over two
    branches (high / low) so that weights sum to one.

    :param in_channels: Number of input channels.
    :param local_window_size: Local window size for spectral aggregation.
    :param use_event_cues: Whether to inject event cues into the fusion gate.
    :param use_semantic_score: Whether to inject semantic score into the gate.
    """

    def __init__(
        self,
        in_channels: int,
        local_window_size: int = 3,
        use_event_cues: bool = True,
        use_semantic_score: bool = True,
    ):
        super().__init__()
        self.use_event_cues = use_event_cues
        self.use_semantic_score = use_semantic_score

        self.spectral_feature_aggregator = EuclideanSpectralFeatureAggregator(
            local_window_size=local_window_size
        )
        self.spatial_detail_enhancer = SpatialDetailEnhancer(
            in_channels=in_channels
        )

        extra_gate_channels = 2  # response_high + response_low
        if use_event_cues:
            extra_gate_channels += 4  # density, gradient, burst, empty_mask
        if use_semantic_score:
            extra_gate_channels += 1

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(extra_gate_channels, 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1, bias=False),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        high_level_spectral_feature: torch.Tensor,
        low_level_spatial_feature: torch.Tensor,
        event_cues: dict[str, torch.Tensor] | None = None,
        semantic_score_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse high-level and low-level features with event guidance.

        :param high_level_spectral_feature: High-level feature map.
        :param low_level_spatial_feature: Low-level feature map.
        :param event_cues: Optional event cues dict.
        :param semantic_score_map: Optional semantic score map [B, 1, H, W].
        :return: Fused feature map.
        """
        aggregated_high = self.spectral_feature_aggregator(
            high_level_spectral_feature, high_level_spectral_feature
        )
        aligned_high = F.interpolate(
            aggregated_high,
            size=low_level_spatial_feature.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        enhanced_low = self.spatial_detail_enhancer(low_level_spatial_feature)

        response_high = aligned_high.mean(dim=1, keepdim=True)
        response_low = enhanced_low.mean(dim=1, keepdim=True)

        fusion_inputs = [response_high, response_low]
        if self.use_event_cues and event_cues is not None:
            fusion_inputs += [
                event_cues["density"],
                event_cues["gradient"],
                event_cues["burst"],
                event_cues["empty_mask"],
            ]
        if self.use_semantic_score:
            if semantic_score_map is not None:
                fusion_inputs += [semantic_score_map]
            else:
                fusion_inputs += [
                    torch.zeros_like(response_high)
                ]

        alpha_logits = self.fusion_gate(torch.cat(fusion_inputs, dim=1))
        alpha = torch.softmax(alpha_logits, dim=1)

        fused = (
            alpha[:, 0:1] * aligned_high + alpha[:, 1:2] * enhanced_low
        )
        out = high_level_spectral_feature + self.res_scale * self.out_proj(
            fused
        )
        return out
