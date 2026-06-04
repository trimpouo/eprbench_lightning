"""Event cue extractor for event-based visual place recognition.

Extracts event-specific cues (density, gradient, burst, empty mask) from
raw event frames to guide token selection and feature modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventCueExtractor(nn.Module):
    """Extract event structure and reliability cues from event frames.

    Operates on unnormalized or roughly [0,1] event frames to compute
    per-patch density, gradient strength, burst noise, and empty mask.

    :param patch_size: Patch size for pooling cues to token grid.
    :param eps: Epsilon for numerical stability.
    """

    def __init__(self, patch_size: int = 14, eps: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer(
            "sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False
        )

    def _minmax_norm(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.flatten(1)
        mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
        mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (x - mn) / (mx - mn + self.eps)

    def forward(
        self, x_raw: torch.Tensor, out_hw: tuple
    ) -> dict[str, torch.Tensor]:
        """Extract event cues.

        :param x_raw: Input event frames [B, 3, H, W]. Expected unnormalized
            or roughly in [0, 1].
        :param out_hw: Target output spatial size (Hf, Wf), e.g. token grid.
        :return: Dictionary with keys ``density``, ``gradient``, ``burst``,
            ``empty_mask``. Each value has shape [B, 1, Hf, Wf].
        """
        # Compatible with RGB-encoded event frames: use absolute intensity
        event_mag = x_raw.float().abs().mean(dim=1, keepdim=True)

        # Patch-level event density
        density = F.avg_pool2d(
            event_mag, kernel_size=self.patch_size, stride=self.patch_size
        )
        density = F.interpolate(
            density, size=out_hw, mode="bilinear", align_corners=False
        )
        density = self._minmax_norm(density)

        # Sobel gradient for structural edges
        gx = F.conv2d(event_mag, self.sobel_x, padding=1)
        gy = F.conv2d(event_mag, self.sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + self.eps)
        grad = F.avg_pool2d(
            grad, kernel_size=self.patch_size, stride=self.patch_size
        )
        grad = F.interpolate(
            grad, size=out_hw, mode="bilinear", align_corners=False
        )
        grad = self._minmax_norm(grad)

        # Burst: local deviation from neighborhood mean
        local_mean = F.avg_pool2d(
            event_mag, kernel_size=7, stride=1, padding=3
        )
        burst = (event_mag - local_mean).abs()
        burst = F.avg_pool2d(
            burst, kernel_size=self.patch_size, stride=self.patch_size
        )
        burst = F.interpolate(
            burst, size=out_hw, mode="bilinear", align_corners=False
        )
        burst = self._minmax_norm(burst)

        # Empty mask: very low density patches are likely background/empty
        empty_mask = (density > 0.03).float()

        return {
            "density": density,
            "gradient": grad,
            "burst": burst,
            "empty_mask": empty_mask,
        }
