import numbers

import torch
import torch.nn as nn


class RMSLayerNormWithBias(nn.Module):
    """RMS-style LayerNorm with learnable scale and shift.

    :param normalized_channels: Channel dimension to normalize over.
    """

    def __init__(self, normalized_channels):
        super().__init__()
        if isinstance(normalized_channels, numbers.Integral):
            normalized_channels = (normalized_channels,)
        normalized_channels = torch.Size(normalized_channels)
        assert len(normalized_channels) == 1

        self.scale = nn.Parameter(torch.ones(normalized_channels))
        self.shift = nn.Parameter(torch.zeros(normalized_channels))
        self.normalized_channels = normalized_channels

    def forward(self, feature_sequence):
        channel_mean_square = feature_sequence.pow(2).mean(-1, keepdim=True)
        normalized_sequence = (
            feature_sequence * torch.rsqrt(channel_mean_square + 1e-6) * self.scale
            + self.shift
        )
        return normalized_sequence


class ImageLayerNorm(nn.Module):
    """LayerNorm for 4-D image features [B, C, H, W].

    :param channels: Number of channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.sequence_norm = RMSLayerNormWithBias(channels)

    def forward(self, feature_map):
        batch, channels, height, width = feature_map.shape
        feature_sequence = feature_map.flatten(2).permute(0, 2, 1)
        normalized_sequence = self.sequence_norm(feature_sequence)
        normalized_feature_map = normalized_sequence.permute(0, 2, 1).view(
            batch, channels, height, width
        )
        return normalized_feature_map


class ChannelAttention(nn.Module):
    """Channel attention module with average and max pooling branches.

    :param channels: Number of input channels.
    :param reduction_ratio: Channel reduction ratio for the MLP.
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
        )
        self.channel_gate = nn.Sigmoid()

    def forward(self, feature_map):
        avg_descriptor = self.shared_channel_mlp(self.global_avg_pool(feature_map))
        max_descriptor = self.shared_channel_mlp(self.global_max_pool(feature_map))
        channel_attention_logits = avg_descriptor + max_descriptor
        channel_attention_map = self.channel_gate(channel_attention_logits)
        return channel_attention_map


class SpatialAttention(nn.Module):
    """Spatial attention module using average and max channel descriptors.

    :param kernel_size: Convolution kernel size for spatial fusion.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial_fusion_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.spatial_gate = nn.Sigmoid()

    def forward(self, feature_map):
        avg_spatial_descriptor = torch.mean(feature_map, dim=1, keepdim=True)
        max_spatial_descriptor, _ = torch.max(feature_map, dim=1, keepdim=True)
        spatial_descriptor = torch.cat(
            [avg_spatial_descriptor, max_spatial_descriptor], dim=1
        )
        spatial_attention_logits = self.spatial_fusion_conv(spatial_descriptor)
        spatial_attention_map = self.spatial_gate(spatial_attention_logits)
        return spatial_attention_map


class SpatialChannelFeatureModulator(nn.Module):
    """Spatial-Channel Feature Modulator (SCFM).

    Adaptively enhances salient spatial locations and important channels
    via parallel spatial and channel attention branches.

    :param channels: Number of input/output channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.input_norm = ImageLayerNorm(channels)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.channel_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.spatial_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input_feature):
        normalized_feature = self.input_norm(input_feature)

        channel_weight = self.channel_attention(normalized_feature)
        channel_modulated_feature = channel_weight * normalized_feature
        channel_branch_output = self.channel_projection(channel_modulated_feature)

        spatial_weight = self.spatial_attention(normalized_feature)
        spatial_modulated_feature = spatial_weight * normalized_feature
        spatial_branch_output = self.spatial_projection(spatial_modulated_feature)

        output_feature = channel_branch_output + spatial_branch_output + input_feature
        return output_feature


class EventReliableSpatialAttention(nn.Module):
    """Spatial attention that optionally ingests event cues.

    :param kernel_size: Convolution kernel size.
    :param use_event_cues: Whether to append event cue channels.
    """

    def __init__(self, kernel_size: int = 7, use_event_cues: bool = True):
        super().__init__()
        self.use_event_cues = use_event_cues
        in_ch = 2
        if use_event_cues:
            in_ch += 4  # density, gradient, 1-burst, empty_mask
        self.spatial_fusion_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.spatial_gate = nn.Sigmoid()

    def forward(
        self,
        feature_map: torch.Tensor,
        event_cues: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        avg_spatial_descriptor = torch.mean(feature_map, dim=1, keepdim=True)
        max_spatial_descriptor, _ = torch.max(feature_map, dim=1, keepdim=True)
        spatial_descriptor = [avg_spatial_descriptor, max_spatial_descriptor]
        if self.use_event_cues and event_cues is not None:
            spatial_descriptor += [
                event_cues["density"],
                event_cues["gradient"],
                1.0 - event_cues["burst"],
                event_cues["empty_mask"],
            ]
        spatial_descriptor = torch.cat(spatial_descriptor, dim=1)
        spatial_attention_logits = self.spatial_fusion_conv(spatial_descriptor)
        spatial_attention_map = self.spatial_gate(spatial_attention_logits)
        return spatial_attention_map


class EventReliableSCFM(nn.Module):
    """Event-Reliable Spatial-Channel Feature Modulator (ER-SCFM).

    Adds event-cue-guided spatial attention and a zero-initialized residual
    scale so that the module starts as an identity mapping.

    :param channels: Number of input/output channels.
    :param use_event_cues: Whether to use event cues in spatial attention.
    """

    def __init__(self, channels: int, use_event_cues: bool = True):
        super().__init__()
        self.input_norm = ImageLayerNorm(channels)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = EventReliableSpatialAttention(
            kernel_size=7, use_event_cues=use_event_cues
        )
        self.channel_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.spatial_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_feature: torch.Tensor,
        event_cues: dict[str, torch.Tensor] | None = None,
        text_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Modulate features with event-aware spatial attention.

        :param input_feature: Input feature map [B, C, H, W].
        :param event_cues: Optional event cues dict.
        :param text_confidence: Unused placeholder for interface consistency.
        :return: Modulated feature map.
        """
        normalized_feature = self.input_norm(input_feature)

        channel_weight = self.channel_attention(normalized_feature)
        channel_modulated_feature = channel_weight * normalized_feature
        channel_branch_output = self.channel_projection(channel_modulated_feature)

        spatial_weight = self.spatial_attention(normalized_feature, event_cues)
        spatial_modulated_feature = spatial_weight * normalized_feature
        spatial_branch_output = self.spatial_projection(spatial_modulated_feature)

        modulated = channel_branch_output + spatial_branch_output
        output_feature = input_feature + self.res_scale * modulated
        return output_feature
