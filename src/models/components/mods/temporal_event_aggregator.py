"""Temporal Event Frame Aggregator (TEFA) for event-based VPR.

Aggregates features from multiple event frames of the same place using
frame-level reliability gating based on event cues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEventFrameAggregator(nn.Module):
    """Aggregate T event frames into a single fused feature map and cls token.

    Uses per-frame statistics (density, non-empty ratio, gradient, burst,
    semantic score) to compute adaptive frame weights.

    :param channels: Feature channel dimension.
    """

    def __init__(self, channels: int = 768):
        super().__init__()
        self.frame_gate = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.patch_gate = nn.Sequential(
            nn.Conv2d(channels + 3, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
        )

    def forward(
        self,
        maps: torch.Tensor,
        cls_tokens: torch.Tensor,
        event_cues_list: list[dict[str, torch.Tensor]] | None = None,
        sem_score_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse temporal event frames.

        :param maps: Patch feature maps [B, T, C, H, W].
        :param cls_tokens: CLS tokens [B, T, C].
        :param event_cues_list: List of T cue dicts, each with keys
            ``density``, ``empty_mask``, ``gradient``, ``burst``.
        :param sem_score_maps: Optional semantic score maps [B, T, 1, H, W]
            or [B, T, H, W].
        :return: Tuple of ``fused_map`` [B, C, H, W], ``fused_cls`` [B, C],
            ``frame_w`` [B, T].
        """
        b, t, c, h, w = maps.shape

        if event_cues_list is None or len(event_cues_list) == 0:
            density_mean = maps.new_zeros(b, t)
            non_empty = maps.new_ones(b, t)
            grad_mean = maps.new_zeros(b, t)
            burst_mean = maps.new_zeros(b, t)
        else:
            density_mean = torch.stack(
                [
                    cues["density"].mean(dim=(1, 2, 3))
                    for cues in event_cues_list
                ],
                dim=1,
            )
            non_empty = torch.stack(
                [
                    cues["empty_mask"].mean(dim=(1, 2, 3))
                    for cues in event_cues_list
                ],
                dim=1,
            )
            grad_mean = torch.stack(
                [
                    cues["gradient"].mean(dim=(1, 2, 3))
                    for cues in event_cues_list
                ],
                dim=1,
            )
            burst_mean = torch.stack(
                [
                    cues["burst"].mean(dim=(1, 2, 3))
                    for cues in event_cues_list
                ],
                dim=1,
            )

        if sem_score_maps is None:
            sem_mean = maps.new_zeros(b, t)
        else:
            if sem_score_maps.dim() == 5:
                sem_mean = sem_score_maps.mean(dim=(2, 3, 4)).squeeze(-1)
            else:
                sem_mean = sem_score_maps.mean(dim=(2, 3))

        stats = torch.stack(
            [
                density_mean,
                non_empty,
                grad_mean,
                1.0 - burst_mean,
                sem_mean,
            ],
            dim=-1,
        )  # [B, T, 5]
        frame_logits = self.frame_gate(stats).squeeze(-1)  # [B, T]
        frame_w = F.softmax(frame_logits, dim=1)

        # Frame-level fusion
        fused_map = (maps * frame_w.view(b, t, 1, 1, 1)).sum(dim=1)
        fused_cls = (cls_tokens * frame_w.view(b, t, 1)).sum(dim=1)

        return fused_map, fused_cls, frame_w
