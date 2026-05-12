"""Frequency Band-Guided Weighting (FBGW) module.

Reference:
    "Statistical Characteristic-Guided Denoising for Rapid High-Resolution
    Transmission Electron Microscopy Imaging", CVPR 2026.

The module transforms spatial features into the frequency domain via RFFT,
applies band-wise attention to enhance signal bands and suppress noise bands,
then transforms back to the spatial domain via IRFFT.
"""

import torch
import torch.nn as nn


class FrequencyBandAttention(nn.Module):
    """Generate per-band importance weights from frequency-domain features.

    Uses both average-pooling and max-pooling global statistics to produce
    channel-wise sigmoid weights that modulate frequency bands.

    :param num_channels: Number of input channels (frequency bands).
    :param reduction_ratio: Channel reduction ratio for the weight MLP.
    """

    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()

        # Local band feature modelling
        self.local_band_conv = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )

        # Global average pooling: per-band average response
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Global max pooling: per-band peak response
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(1, num_channels // reduction_ratio)

        # Band weight generation network
        self.band_weight_mlp = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, frequency_band_feature):
        """Forward pass.

        :param frequency_band_feature: Tensor of shape ``(B, C, H, W)``.
        :return: Band importance weights of shape ``(B, C, 1, 1)``.
        """
        # Local band feature modelling
        refined_band_feature = self.local_band_conv(frequency_band_feature)

        # Average-pooling branch
        avg_band_weight = self.band_weight_mlp(
            self.global_avg_pool(refined_band_feature)
        )

        # Max-pooling branch
        max_band_weight = self.band_weight_mlp(
            self.global_max_pool(refined_band_feature)
        )

        # Combine both global statistics
        frequency_band_weight = avg_band_weight + max_band_weight
        return frequency_band_weight


class FrequencyBandGuidedUnit(nn.Module):
    """Frequency Band-Guided Weighting unit.

    Pipeline:
        1. RFFT: spatial features -> frequency-domain complex features.
        2. Split complex features into real and imaginary parts.
        3. Concatenate band position encodings.
        4. 1x1 conv for band decoupling.
        5. Band attention to enhance signal bands / suppress noise bands.
        6. 1x1 conv for band re-coupling.
        7. IRFFT: frequency domain -> spatial features.

    :param in_channels: Number of input spatial channels.
    :param out_channels: Number of output spatial channels.
    :param reduction_ratio: Channel reduction ratio for band attention.
    """

    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super().__init__()

        # Band decoupling conv: 2C (real+imag) + 2 (position enc) -> 2*out_channels
        self.band_decoupling_conv = nn.Conv2d(
            in_channels=in_channels * 2 + 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Frequency-domain normalisation and activation
        self.band_norm = nn.BatchNorm2d(out_channels * 2)
        self.band_activation = nn.ReLU(inplace=True)

        # Band attention
        self.frequency_band_attention = FrequencyBandAttention(
            out_channels * 2, reduction_ratio=reduction_ratio
        )

        # Band re-coupling conv
        self.band_coupling_conv = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, spatial_feature):
        """Forward pass.

        :param spatial_feature: Tensor of shape ``(B, C, H, W)``.
        :return: Enhanced spatial feature of shape ``(B, out_channels, H, W)``.
        """
        batch_size = spatial_feature.shape[0]
        fft_dims = (-2, -1)
        original_spatial_size = spatial_feature.shape[-2:]

        # RFFT: spatial -> frequency
        complex_frequency_feature = torch.fft.rfftn(
            spatial_feature, dim=fft_dims, norm="ortho"
        )

        # Split complex into real & imaginary, then merge to channel dim
        real_imag_frequency_feature = torch.stack(
            (complex_frequency_feature.real, complex_frequency_feature.imag), dim=-1
        )
        # [B, C, H, W_freq, 2] -> [B, C, 2, H, W_freq]
        real_imag_frequency_feature = real_imag_frequency_feature.permute(
            0, 1, 4, 2, 3
        ).contiguous()
        # [B, 2C, H, W_freq]
        frequency_content_feature = real_imag_frequency_feature.view(
            (batch_size, -1) + real_imag_frequency_feature.size()[3:]
        )
        frequency_height, frequency_width = frequency_content_feature.shape[-2:]

        # Band position encodings
        vertical_band_position = (
            torch.linspace(0, 1, frequency_height)[None, None, :, None]
            .expand(batch_size, 1, frequency_height, frequency_width)
            .to(frequency_content_feature)
        )
        horizontal_band_position = (
            torch.linspace(0, 1, frequency_width)[None, None, None, :]
            .expand(batch_size, 1, frequency_height, frequency_width)
            .to(frequency_content_feature)
        )
        band_characteristic_feature = torch.cat(
            (vertical_band_position, horizontal_band_position, frequency_content_feature),
            dim=1,
        )

        # Band decoupling
        decoupled_band_feature = self.band_decoupling_conv(band_characteristic_feature)
        decoupled_band_feature = self.band_activation(
            self.band_norm(decoupled_band_feature)
        )

        # Band attention weighting
        band_importance_weight = self.frequency_band_attention(decoupled_band_feature)
        weighted_band_feature = decoupled_band_feature * band_importance_weight

        # Band re-coupling
        recoupled_frequency_feature = self.band_coupling_conv(weighted_band_feature)
        # Split channels back into real & imaginary
        recoupled_frequency_feature = recoupled_frequency_feature.view(
            (batch_size, -1, 2) + recoupled_frequency_feature.size()[2:]
        )
        # [B, C, 2, H, W_freq] -> [B, C, H, W_freq, 2]
        recoupled_frequency_feature = recoupled_frequency_feature.permute(
            0, 1, 3, 4, 2
        ).contiguous()
        # Reconstruct complex tensor
        complex_recoupled_frequency_feature = torch.complex(
            recoupled_frequency_feature[..., 0], recoupled_frequency_feature[..., 1]
        )

        # IRFFT: frequency -> spatial
        enhanced_spatial_feature = torch.fft.irfftn(
            complex_recoupled_frequency_feature,
            s=original_spatial_size,
            dim=fft_dims,
            norm="ortho",
        )

        return enhanced_spatial_feature