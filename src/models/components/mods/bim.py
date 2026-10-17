"""Brain-Inspired Module (BIM).

Paper: "Visual-in-Visual: A Unified and Efficient Baseline for Image Restoration"
       (TPAMI 2026, https://ieeexplore.ieee.org/abstract/document/11419859/)

BIM mimics the human visual processing pipeline through multi-scale convolutional
encoding, similarity-aware weighting, and high-order feature interaction.  It
enhances informative features while suppressing redundant responses in a
lightweight fashion.

:param channels: Number of input/output channels (must match the feature-map
    channel dimension).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainInspiredModule(nn.Module):
    """Brain-Inspired Module that enhances features via multi-scale encoding,
    similarity-aware channel attention, and high-order feature interaction.

    :param channels: Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Local branch: 1x1 projection + 3x3 depthwise conv for local texture
        self.local_branch_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.local_depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # Context branch: 1x1 projection + 9x9 depthwise conv for large-range info
        self.context_branch_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.context_depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=9, padding=4, groups=channels
        )

        # Similarity-weight MLP: maps 2C-channel similarity to scalar weight
        hidden_channels = int(1.2 * channels)
        self.similarity_weight_mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

        # High-order interaction depthwise convolutions
        self.high_order_depthwise_conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.high_order_depthwise_conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # Output projection
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input feature map of shape ``[B, C, H, W]``.
        :return: Enhanced feature map of the same shape.
        """
        batch_size, channels, height, width = x.shape

        # Multi-scale feature extraction
        local_feature = self.local_depthwise_conv(self.local_branch_proj(x))
        context_feature = self.context_depthwise_conv(self.context_branch_proj(x))

        # Similarity-aware channel attention
        multi_scale_feature = torch.cat([local_feature, context_feature], dim=1)
        flattened_feature = multi_scale_feature.view(batch_size, 2 * channels, -1)
        normalized_feature = F.normalize(flattened_feature, p=2, dim=-1)
        channel_similarity_matrix = torch.bmm(
            normalized_feature, normalized_feature.transpose(1, 2)
        )
        channel_attention = self.similarity_weight_mlp(channel_similarity_matrix)
        channel_attention = channel_attention.view(batch_size, 2 * channels, 1, 1)

        local_attention, context_attention = torch.split(
            channel_attention, channels, dim=1
        )

        weighted_local_feature = local_feature * local_attention
        weighted_context_feature = context_feature * context_attention

        # High-order feature interaction
        first_order_feature = weighted_local_feature * weighted_context_feature
        second_order_feature = (
            self.high_order_depthwise_conv1(first_order_feature)
            * weighted_context_feature
        )
        third_order_feature = (
            self.high_order_depthwise_conv2(second_order_feature)
            * weighted_local_feature
        )

        output = self.output_proj(third_order_feature)
        return output