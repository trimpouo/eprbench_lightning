"""Low-frequency Guided Feature Purification (LFP) module.

Reference:
    Yuan et al., "Seeing Through the Noise: Improving Infrared Small Target
    Detection and Segmentation from Noise Suppression Perspective", CVPR 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:

    class DWTForward(nn.Module):
        """Pure-PyTorch 1-level 2D Haar DWT fallback when pytorch_wavelets is absent."""

        def __init__(self, J: int = 1, wave: str = "haar", mode: str = "zero") -> None:
            super().__init__()
            self.J = J
            self.wave = wave
            self.mode = mode

        def forward(self, x: torch.Tensor):
            """Compute 1-level 2D Haar wavelet decomposition."""
            if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
                x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="reflect")
            x00 = x[:, :, 0::2, 0::2]
            x01 = x[:, :, 0::2, 1::2]
            x10 = x[:, :, 1::2, 0::2]
            x11 = x[:, :, 1::2, 1::2]
            ll = (x00 + x01 + x10 + x11) * 0.5
            lh = (x00 + x01 - x10 - x11) * 0.5
            hl = (x00 - x01 + x10 - x11) * 0.5
            hh = (x00 - x01 - x10 + x11) * 0.5
            high = torch.stack([lh, hl, hh], dim=2)
            return ll, [high]

    class DWTInverse(nn.Module):
        """Pure-PyTorch 1-level 2D Haar IDWT fallback when pytorch_wavelets is absent."""

        def __init__(self, wave: str = "haar", mode: str = "zero") -> None:
            super().__init__()
            self.wave = wave
            self.mode = mode

        def forward(self, coeffs):
            """Reconstruct 2D feature map from (LL, [high]) Haar coefficients."""
            ll, high_list = coeffs
            high = high_list[0]
            lh, hl, hh = high[:, :, 0], high[:, :, 1], high[:, :, 2]
            b, c, h, w = ll.shape
            out = ll.new_zeros((b, c, h * 2, w * 2))
            out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
            out[:, :, 0::2, 1::2] = (ll + lh - hl - hh) * 0.5
            out[:, :, 1::2, 0::2] = (ll - lh + hl - hh) * 0.5
            out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
            return out


class DiscreteWaveletTransform2D(nn.Module):
    """Single-level 2D discrete wavelet forward transform.

    Decomposes an input feature into low-frequency LL and three high-frequency
    sub-bands LH, HL, HH.

    :param wavelet_type: Wavelet family (e.g. ``'haar'``).
    :param padding_mode: Padding mode for DWT.
    """

    def __init__(self, wavelet_type="haar", padding_mode="zero"):
        super().__init__()
        self.forward_wavelet_transform = DWTForward(
            J=1, wave=wavelet_type, mode=padding_mode
        )

    def forward(self, input_feature):
        """Forward pass.

        :param input_feature: Tensor of shape ``(B, C, H, W)``.
        :return: Concatenated wavelet features of shape ``(B, 4*C, H/2, W/2)``.
        """
        b, c, h, w = input_feature.shape

        with torch.cuda.amp.autocast(enabled=False):
            if input_feature.dtype != torch.float32:
                input_feature = input_feature.float()
            low_freq, high_freq_list = self.forward_wavelet_transform(input_feature)

        # high_freq_list[0]: (B, C, 3, H/2, W/2)
        high_freq = (
            high_freq_list[0]
            .transpose(1, 2)
            .reshape(b, -1, high_freq_list[0].shape[3], high_freq_list[0].shape[4])
        )

        wavelet_feature = torch.cat((low_freq, high_freq), dim=1)
        wavelet_feature = F.interpolate(
            wavelet_feature, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )
        return wavelet_feature


class InverseDiscreteWaveletTransform2D(nn.Module):
    """Single-level 2D discrete wavelet inverse transform.

    Reconstructs a spatial-domain feature from low- and high-frequency sub-bands.

    :param wavelet_type: Wavelet family.
    :param padding_mode: Padding mode for IDWT.
    """

    def __init__(self, wavelet_type="haar", padding_mode="zero"):
        super().__init__()
        self.inverse_wavelet_transform = DWTInverse(
            wave=wavelet_type, mode=padding_mode
        )

    def forward(self, low_frequency_feature, high_frequency_feature):
        """Forward pass.

        :param low_frequency_feature: Tensor of shape ``(B, C, H/2, W/2)``.
        :param high_frequency_feature: Tensor of shape ``(B, 3*C, H/2, W/2)``.
        :return: Reconstructed feature of shape ``(B, C, H, W)``.
        """
        b, c, hh, hw = low_frequency_feature.shape

        high_frequency_feature = high_frequency_feature.reshape(
            b, c, 3, hh, hw
        )

        with torch.cuda.amp.autocast(enabled=False):
            reconstructed = self.inverse_wavelet_transform(
                (low_frequency_feature, [high_frequency_feature.float()])
            )

        reconstructed = F.interpolate(
            reconstructed,
            size=(2 * hh, 2 * hw),
            mode="bilinear",
            align_corners=False,
        )
        return reconstructed


class SpatialAttentionMapGenerator(nn.Module):
    """Spatial attention map generator.

    Produces a single-channel spatial weight map via channel-wise pooling
    followed by a convolution.

    :param kernel_size: Convolution kernel size (``3`` or ``7``).
    :param use_bn_before_sigmoid: Whether to apply batch norm before sigmoid.
    """

    def __init__(self, kernel_size=7, use_bn_before_sigmoid=False):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.use_bn_before_sigmoid = use_bn_before_sigmoid

        self.spatial_context_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        if use_bn_before_sigmoid:
            self.attention_bn = nn.BatchNorm2d(1)
            self.attention_bn.bias.data.fill_(0)
            self.attention_bn.bias.requires_grad = False

        self.activation = nn.Sigmoid()

    def forward(self, input_feature):
        """Forward pass.

        :param input_feature: Tensor of shape ``(B, C, H, W)``.
        :return: Spatial attention map of shape ``(B, 1, H, W)``.
        """
        avg_response = torch.mean(input_feature, dim=1, keepdim=True)
        max_response, _ = torch.max(input_feature, dim=1, keepdim=True)

        spatial_descriptor = torch.cat([avg_response, max_response], dim=1)
        attention_logit = self.spatial_context_conv(spatial_descriptor)

        if self.use_bn_before_sigmoid:
            attention_logit = self.attention_bn(attention_logit)

        return self.activation(attention_logit)


class LearnableGaussianSmoothingBank(nn.Module):
    """Learnable Gaussian smoothing filter bank.

    Generates a set of Gaussian kernels with learnable standard deviations
    and applies depth-wise convolutions.

    :param kernel_size: Spatial size of the Gaussian kernel.
    :param num_filters: Number of learnable Gaussian filters.
    :param num_channels: Number of input channels (for grouped convolution).
    """

    def __init__(self, kernel_size, num_filters, num_channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.padding_size = kernel_size // 2

        self.learnable_sigmas = nn.ParameterList(
            [nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
             for _ in range(num_filters)]
        )

    def forward(self, input_feature):
        """Forward pass.

        :param input_feature: Tensor of shape ``(B, C, H, W)``.
        :return: Smoothed feature of shape ``(B, num_filters*C, H, W)``.
        """
        kernels = [
            self._build_gaussian_kernel(
                kernel_size=self.kernel_size, sigma=sigma
            ).repeat(self.num_channels, 1, 1, 1)
            for sigma in self.learnable_sigmas
        ]

        padded = F.pad(
            input_feature,
            (self.padding_size,) * 4,
            mode="replicate",
        )

        smoothed_list = [
            F.conv2d(
                padded,
                weight=kernel.to(input_feature.device),
                groups=self.num_channels,
            )
            for kernel in kernels
        ]

        return torch.cat(smoothed_list, dim=1)

    def _build_gaussian_kernel(self, kernel_size, sigma):
        """Build a 2D Gaussian kernel.

        :param kernel_size: Size of the square kernel.
        :param sigma: Standard deviation (learnable scalar).
        :return: Normalized Gaussian kernel of shape ``(1, 1, K, K)``.
        """
        device = sigma.device
        dtype = sigma.dtype
        kernel = torch.zeros(
            1, 1, kernel_size, kernel_size, dtype=dtype, device=device
        )
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[:, :, i, j] = torch.exp(
                    -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)
                )

        kernel = kernel / kernel.sum()
        return kernel


class LowFrequencyGuidedFeaturePurification(nn.Module):
    """Low-frequency Guided Feature Purification (LFP) module.

    The module decomposes input features via DWT, uses low-frequency
    components to generate a spatial attention map that modulates high-
    frequency features, and optionally applies gated Gaussian smoothing
    to suppress noisy high-frequency responses before reconstructing
    the feature via IDWT.

    :param in_channels: Number of input channels.
    :param wavelet_type: Wavelet family for DWT/IDWT.
    :param padding_mode: Padding mode for wavelet transforms.
    :param enable_gaussian_smoothing: Whether to apply learnable Gaussian
        smoothing on low-confidence high-frequency components.
    :param high_frequency_threshold: Threshold for the gated Gaussian
        smoothing; responses with absolute value below this are smoothed.
    """

    def __init__(
        self,
        in_channels,
        wavelet_type="haar",
        padding_mode="symmetric",
        enable_gaussian_smoothing=True,
        high_frequency_threshold=0.5,
    ):
        super().__init__()
        self.wavelet_decomposition = DiscreteWaveletTransform2D(
            wavelet_type=wavelet_type, padding_mode=padding_mode
        )
        self.wavelet_reconstruction = InverseDiscreteWaveletTransform2D(
            wavelet_type=wavelet_type, padding_mode=padding_mode
        )

        self.enable_gaussian_smoothing = enable_gaussian_smoothing
        self.high_frequency_threshold = high_frequency_threshold

        self.low_frequency_spatial_attention = SpatialAttentionMapGenerator()

        if self.enable_gaussian_smoothing:
            self.high_frequency_gaussian_smoothing = LearnableGaussianSmoothingBank(
                kernel_size=3, num_filters=1, num_channels=3 * in_channels
            )

    def forward(self, input_feature):
        """Forward pass.

        :param input_feature: Tensor of shape ``(B, C, H, W)``.
        :return: Purified feature of shape ``(B, C, H, W)``.
        """
        b, c, h, w = input_feature.shape

        # DWT -> (B, 4*C, H/2, W/2)
        wavelet_feature = self.wavelet_decomposition(input_feature)

        low_frequency_feature = wavelet_feature[:, :c, :, :]
        high_frequency_feature = wavelet_feature[:, c:, :, :]

        # Spatial attention from low-frequency components
        target_spatial_weight_map = self.low_frequency_spatial_attention(
            low_frequency_feature
        )
        modulated_high_frequency = high_frequency_feature * target_spatial_weight_map

        if self.enable_gaussian_smoothing:
            smoothed_high_frequency = self.high_frequency_gaussian_smoothing(
                modulated_high_frequency
            )

            low_confidence_mask = (
                modulated_high_frequency.abs() < self.high_frequency_threshold
            ).float()

            purified_high_frequency = (
                modulated_high_frequency * (1.0 - low_confidence_mask)
                + smoothed_high_frequency * low_confidence_mask
            )
        else:
            purified_high_frequency = modulated_high_frequency

        # IDWT reconstruction
        purified_feature = self.wavelet_reconstruction(
            low_frequency_feature, purified_high_frequency
        )
        return purified_feature
