"""Spectral-guided Adaptive Cross-layer Fusion (SACF) module.

Reference:
    MODA: The First Challenging Benchmark for Multispectral Object
    Detection in Aerial Images (AAAI 2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_prior(
    prior: torch.Tensor | None,
    reference: torch.Tensor,
    default_value: float,
) -> torch.Tensor:
    """Return a single-channel prior aligned with ``reference``.

    :param prior: Optional prior map with shape [B, 1, H, W] or [B, H, W].
    :param reference: Reference feature map [B, C, H, W].
    :param default_value: Fill value used when ``prior`` is missing.
    :return: Prior map [B, 1, H, W].
    """
    batch_size, _, height, width = reference.shape
    if prior is None:
        return reference.new_full((batch_size, 1, height, width), default_value)

    if prior.dim() == 3:
        prior = prior.unsqueeze(1)

    prior = prior.to(device=reference.device, dtype=reference.dtype)
    if prior.shape[-2:] != (height, width):
        prior = F.interpolate(
            prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    return prior.clamp(0.0, 1.0)


class RBFKANLayer(nn.Module):
    """Lightweight RBF-KAN layer used for small gating functions.

    The layer keeps a regular linear base branch and adds a Gaussian radial
    basis expansion branch. It is intentionally compact so that SACF gates can
    use KAN-style nonlinear edge functions without adding an external package.

    :param in_dim: Input feature dimension.
    :param out_dim: Output feature dimension.
    :param num_grids: Number of radial basis centers per input dimension.
    :param grid_min: Minimum grid center.
    :param grid_max: Maximum grid center.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_grids: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
    ):
        super().__init__()
        if num_grids < 2:
            raise ValueError("num_grids must be at least 2 for RBFKANLayer.")

        self.in_dim = in_dim
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid.view(1, 1, num_grids), persistent=False)
        self.grid_step = float(grid[1] - grid[0])

        self.base_activation = nn.SiLU()
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.rbf_linear = nn.Linear(in_dim * num_grids, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base and RBF-expanded projections.

        :param x: Input tensor [..., in_dim].
        :return: Output tensor [..., out_dim].
        """
        basis = torch.exp(-((x.unsqueeze(-1) - self.grid) / self.grid_step) ** 2)
        basis = basis.flatten(start_dim=-2)
        return self.base_linear(self.base_activation(x)) + self.rbf_linear(basis)


class FastKAN(nn.Module):
    """Small FastKAN-style network built from Gaussian RBF-KAN layers.

    :param layers: Layer dimensions, e.g. ``[4, 16, 3]``.
    :param num_grids: Number of radial basis centers per input dimension.
    :param dropout: Dropout probability between hidden layers.
    """

    def __init__(
        self,
        layers: list[int],
        num_grids: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("FastKAN needs at least input and output dimensions.")

        modules: list[nn.Module] = []
        for idx in range(len(layers) - 1):
            modules.append(
                RBFKANLayer(
                    in_dim=layers[idx],
                    out_dim=layers[idx + 1],
                    num_grids=num_grids,
                )
            )
            if idx < len(layers) - 2:
                modules.append(nn.SiLU())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the KAN network.

        :param x: Input tensor [..., in_dim].
        :return: Output tensor [..., out_dim].
        """
        return self.net(x)


class FastKANGate(nn.Module):
    """Token-wise FastKAN gate for spatial maps.

    :param in_dim: Number of gate input channels.
    :param hidden: Hidden dimension of the KAN gate.
    :param out_dim: Number of gate output branches.
    :param num_grids: Number of RBF grid centers.
    :param dropout: Dropout probability inside the KAN gate.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden: int = 16,
        out_dim: int = 3,
        num_grids: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.kan = FastKAN(
            layers=[in_dim, hidden, out_dim],
            num_grids=num_grids,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict branch logits from a gate feature map.

        :param x: Gate input [B, C, H, W].
        :return: Gate logits [B, out_dim, H, W].
        """
        batch_size, channels, height, width = x.shape
        z = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        z = self.norm(z)
        z = self.kan(z)
        z = z.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
        return z


class CosineSpectralFeatureAggregator(nn.Module):
    """Local cosine feature aggregation with semantic and event priors.

    :param in_channels: Number of feature channels.
    :param local_window_size: Size of the local aggregation window.
    :param query_channels: Query/key channels. Defaults to ``in_channels // 4``.
    :param tau: Softmax temperature for local cosine scores.
    :param lambda_sem: Weight of semantic prior bias.
    :param lambda_evt: Weight of event reliability prior bias.
    :param dilations: Dilation values for multi-scale local aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        local_window_size: int = 3,
        query_channels: int | None = None,
        tau: float = 0.1,
        lambda_sem: float = 0.3,
        lambda_evt: float = 0.2,
        dilations: tuple[int, ...] = (1,),
    ):
        super().__init__()
        self.local_window_size = local_window_size
        self.tau = tau
        self.lambda_sem = lambda_sem
        self.lambda_evt = lambda_evt
        self.dilations = tuple(int(d) for d in dilations)

        if len(self.dilations) == 0:
            raise ValueError("At least one dilation is required.")

        query_channels = query_channels or max(1, in_channels // 4)
        self.q = nn.Conv2d(in_channels, query_channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(in_channels, query_channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.unfolds = nn.ModuleList(
            [
                nn.Unfold(kernel_size=local_window_size, dilation=dilation)
                for dilation in self.dilations
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.dilations)))

    def _unfold_with_replicate_pad(
        self,
        x: torch.Tensor,
        unfold: nn.Unfold,
        dilation: int,
    ) -> torch.Tensor:
        padding = dilation * (self.local_window_size - 1) // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        return unfold(x)

    def _aggregate_one_scale(
        self,
        center_feature: torch.Tensor,
        reference_feature: torch.Tensor,
        semantic_score_map: torch.Tensor,
        event_reliability_map: torch.Tensor,
        unfold: nn.Unfold,
        dilation: int,
    ) -> torch.Tensor:
        batch_size, channels, height, width = center_feature.shape
        num_neighbors = self.local_window_size * self.local_window_size
        num_positions = height * width

        q_map = F.normalize(self.q(center_feature), dim=1)
        k_map = F.normalize(self.k(reference_feature), dim=1)
        v_map = self.v(reference_feature)

        q_vectors = q_map.flatten(2).transpose(1, 2).unsqueeze(2)
        local_k = self._unfold_with_replicate_pad(k_map, unfold, dilation)
        local_k = local_k.reshape(
            batch_size, -1, num_neighbors, num_positions
        ).permute(0, 3, 2, 1)
        local_v = self._unfold_with_replicate_pad(v_map, unfold, dilation)
        local_v = local_v.reshape(
            batch_size, channels, num_neighbors, num_positions
        ).permute(0, 3, 2, 1)

        score = (local_k * q_vectors).sum(dim=-1) / max(self.tau, 1e-6)

        semantic_prior = self._unfold_with_replicate_pad(
            semantic_score_map, unfold, dilation
        )
        semantic_prior = semantic_prior.reshape(
            batch_size, num_neighbors, num_positions
        ).transpose(1, 2)
        event_prior = self._unfold_with_replicate_pad(
            event_reliability_map, unfold, dilation
        )
        event_prior = event_prior.reshape(
            batch_size, num_neighbors, num_positions
        ).transpose(1, 2)

        score = (
            score
            + self.lambda_sem * semantic_prior
            + self.lambda_evt * event_prior
        )
        weights = F.softmax(score, dim=-1)
        aggregated = (weights.unsqueeze(-1) * local_v).sum(dim=2)
        aggregated = aggregated.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        return aggregated

    def forward(
        self,
        center_feature: torch.Tensor,
        reference_feature: torch.Tensor,
        semantic_score_map: torch.Tensor | None = None,
        event_reliability_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate local neighborhoods around each high-level token.

        :param center_feature: Center feature map [B, C, H, W].
        :param reference_feature: Reference feature map [B, C, H, W].
        :param semantic_score_map: Semantic prior [B, 1, H, W].
        :param event_reliability_map: Event reliability prior [B, 1, H, W].
        :return: Aggregated high-level feature map.
        """
        semantic_score_map = _resize_prior(
            semantic_score_map,
            center_feature,
            default_value=0.0,
        )
        event_reliability_map = _resize_prior(
            event_reliability_map,
            center_feature,
            default_value=1.0,
        )

        outputs = [
            self._aggregate_one_scale(
                center_feature,
                reference_feature,
                semantic_score_map,
                event_reliability_map,
                unfold,
                dilation,
            )
            for unfold, dilation in zip(self.unfolds, self.dilations)
        ]

        if len(outputs) == 1:
            return outputs[0]

        scale_weight = torch.softmax(self.scale_logits, dim=0)
        output = torch.zeros_like(outputs[0])
        for weight, aggregated in zip(scale_weight, outputs):
            output = output + weight.view(1, 1, 1, 1) * aggregated
        return output


class SemanticEventSpatialDetailEnhancer(nn.Module):
    """Enhance low-level details constrained by semantic and event priors.

    :param in_channels: Number of feature channels.
    :param hidden_channels: Hidden channels for detail attention.
    """

    def __init__(self, in_channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden_channels = hidden_channels or max(16, in_channels // 4)
        self.detail_attention = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.low_refine = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(
        self,
        spatial_feature: torch.Tensor,
        semantic_score_map: torch.Tensor | None = None,
        event_reliability_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return stable high-frequency detail.

        :param spatial_feature: Low-level feature map [B, C, H, W].
        :param semantic_score_map: Semantic prior map [B, 1, H, W].
        :param event_reliability_map: Event reliability prior map [B, 1, H, W].
        :return: Refined detail feature [B, C, H, W].
        """
        semantic_score_map = _resize_prior(
            semantic_score_map,
            spatial_feature,
            default_value=0.0,
        )
        event_reliability_map = _resize_prior(
            event_reliability_map,
            spatial_feature,
            default_value=1.0,
        )

        lowpass = F.avg_pool2d(spatial_feature, kernel_size=3, stride=1, padding=1)
        high_frequency_detail = spatial_feature - lowpass
        detail_input = torch.cat(
            [high_frequency_detail, semantic_score_map, event_reliability_map],
            dim=1,
        )
        detail_attention = self.detail_attention(detail_input)
        stable_detail = (
            high_frequency_detail
            * detail_attention
            * event_reliability_map
            * (0.5 + 0.5 * semantic_score_map)
        )
        return self.low_refine(stable_detail)


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


class SemanticEventResidualSACF(nn.Module):
    """Semantic-Event Residual SACF (SE-RSACF).

    The module uses final DINOv2 patch tokens as high-level semantics and
    intermediate tokens as low-level details. Semantic and event priors guide
    both local cosine aggregation and high-frequency detail injection. The
    output is residual so that the pretrained visual/text feature space remains
    the starting point.

    :param in_channels: Number of feature channels.
    :param local_window_size: Local aggregation window size.
    :param tau: Temperature for local cosine aggregation.
    :param lambda_sem: Semantic prior bias weight.
    :param lambda_evt: Event reliability prior bias weight.
    :param dilations: Dilation values for multi-scale local aggregation.
    :param gate_type: ``"conv"`` or ``"fastkan"``.
    :param gate_hidden: Hidden dimension for the gate.
    :param gate_num_grids: Number of RBF grids for FastKAN gate.
    :param gate_dropout: Dropout probability for FastKAN gate.
    :param residual_scale_init: Initial residual scale.
    :param zero_init_out_proj: Whether to zero-initialize residual projection.
    :param use_semantic_prior: Whether to use semantic score prior.
    :param use_event_prior: Whether to use event reliability prior.
    :param event_low_thr: Lower density threshold for event reliability.
    :param event_high_thr: Upper density threshold for event reliability.
    :param event_smooth: Smoothness of density reliability thresholds.
    """

    def __init__(
        self,
        in_channels: int,
        local_window_size: int = 3,
        tau: float = 0.1,
        lambda_sem: float = 0.3,
        lambda_evt: float = 0.2,
        dilations: tuple[int, ...] = (1,),
        gate_type: str = "conv",
        gate_hidden: int = 16,
        gate_num_grids: int = 8,
        gate_dropout: float = 0.0,
        residual_scale_init: float = 0.1,
        zero_init_out_proj: bool = True,
        use_semantic_prior: bool = True,
        use_event_prior: bool = True,
        event_low_thr: float = 0.03,
        event_high_thr: float = 0.95,
        event_smooth: float = 0.10,
    ):
        super().__init__()
        self.use_semantic_prior = use_semantic_prior
        self.use_event_prior = use_event_prior
        self.event_low_thr = event_low_thr
        self.event_high_thr = event_high_thr
        self.event_smooth = event_smooth

        self.spectral_feature_aggregator = CosineSpectralFeatureAggregator(
            in_channels=in_channels,
            local_window_size=local_window_size,
            tau=tau,
            lambda_sem=lambda_sem,
            lambda_evt=lambda_evt,
            dilations=dilations,
        )
        self.spatial_detail_enhancer = SemanticEventSpatialDetailEnhancer(
            in_channels=in_channels
        )

        if gate_type == "fastkan":
            self.fusion_gate = FastKANGate(
                in_dim=4,
                hidden=gate_hidden,
                out_dim=3,
                num_grids=gate_num_grids,
                dropout=gate_dropout,
            )
        elif gate_type == "conv":
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(4, gate_hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(gate_hidden, 3, kernel_size=1, bias=True),
            )
        else:
            raise ValueError(f"Unsupported SE-RSACF gate_type: {gate_type}")

        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        if zero_init_out_proj:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _event_reliability_map(
        self,
        event_cues: dict[str, torch.Tensor] | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_event_prior or event_cues is None:
            return _resize_prior(None, reference, default_value=1.0)

        density = _resize_prior(event_cues.get("density"), reference, 0.0)
        gradient = _resize_prior(event_cues.get("gradient"), reference, 0.0)
        burst = _resize_prior(event_cues.get("burst"), reference, 0.0)
        empty_mask = _resize_prior(event_cues.get("empty_mask"), reference, 1.0)

        smooth = max(self.event_smooth, 1e-6)
        density_reliability = torch.sigmoid(
            (density - self.event_low_thr) / smooth
        ) * torch.sigmoid((self.event_high_thr - density) / smooth)
        structure_reliability = 0.5 + 0.5 * gradient
        burst_reliability = (1.0 - 0.5 * burst).clamp(0.0, 1.0)

        reliability = (
            density_reliability
            * structure_reliability
            * burst_reliability
            * empty_mask
        )
        return reliability.clamp(0.0, 1.0)

    def forward(
        self,
        high_level_spectral_feature: torch.Tensor,
        low_level_spatial_feature: torch.Tensor,
        event_cues: dict[str, torch.Tensor] | None = None,
        semantic_score_map: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse high and low DINOv2 maps with semantic/event residual guidance.

        :param high_level_spectral_feature: Final-layer patch map [B, C, H, W].
        :param low_level_spatial_feature: Intermediate-layer patch map [B, C, H, W].
        :param event_cues: Optional event cue dict from ``EventCueExtractor``.
        :param semantic_score_map: Semantic score map from original high tokens.
        :param return_aux: Whether to return gate and regularization tensors.
        :return: Fused feature map, optionally with auxiliary outputs.
        """
        if low_level_spatial_feature.shape[-2:] != high_level_spectral_feature.shape[-2:]:
            low_level_spatial_feature = F.interpolate(
                low_level_spatial_feature,
                size=high_level_spectral_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.use_semantic_prior:
            semantic_prior = _resize_prior(
                semantic_score_map,
                high_level_spectral_feature,
                default_value=0.0,
            )
        else:
            semantic_prior = _resize_prior(
                None,
                high_level_spectral_feature,
                default_value=0.0,
            )
        event_prior = self._event_reliability_map(
            event_cues,
            high_level_spectral_feature,
        )

        aggregated_high = self.spectral_feature_aggregator(
            high_level_spectral_feature,
            high_level_spectral_feature,
            semantic_score_map=semantic_prior,
            event_reliability_map=event_prior,
        )
        stable_low_detail = self.spatial_detail_enhancer(
            low_level_spatial_feature,
            semantic_score_map=semantic_prior,
            event_reliability_map=event_prior,
        )

        response_high = aggregated_high.mean(dim=1, keepdim=True)
        response_low = stable_low_detail.mean(dim=1, keepdim=True)
        gate_input = torch.cat(
            [response_high, response_low, semantic_prior, event_prior],
            dim=1,
        )
        gate_logits = self.fusion_gate(gate_input)
        gate_weight = torch.softmax(gate_logits, dim=1)

        gate_id = gate_weight[:, 0:1]
        gate_high = gate_weight[:, 1:2]
        gate_low = gate_weight[:, 2:3]
        delta = gate_high * (
            aggregated_high - high_level_spectral_feature
        ) + gate_low * stable_low_detail
        residual = self.res_scale * self.out_proj(delta)
        output = high_level_spectral_feature + residual

        if not return_aux:
            return output

        high_norm = high_level_spectral_feature.float().flatten(1).norm(dim=1)
        residual_norm = residual.float().flatten(1).norm(dim=1)
        residual_loss = (residual_norm / (high_norm + 1e-6)).mean()
        noise_loss = ((1.0 - event_prior) * gate_low).mean()
        aux = {
            "gate_id": gate_id,
            "gate_high": gate_high,
            "gate_low": gate_low,
            "semantic_score_map": semantic_prior,
            "event_reliability_map": event_prior,
            "residual_loss": residual_loss,
            "noise_loss": noise_loss,
            "res_scale": self.res_scale.detach(),
        }
        return output, aux
