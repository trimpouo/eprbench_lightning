"""Event-Aware Token Selector (EATS) for event-based VPR.

Replaces fixed text-only top-k patch selection with a dynamic scoring
mechanism that incorporates event density, structural gradient, burst
noise penalty, and empty-patch suppression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventAwareTokenSelector(nn.Module):
    """Dynamic token selector guided by both semantic similarity and event cues.

    Computes per-patch scores combining text-patch similarity, event structure
    score, and burst noise penalty. Dynamically adjusts the keep ratio ``rho``
    based on score distribution statistics.

    :param rho_min: Minimum keep ratio.
    :param rho_max: Maximum keep ratio.
    :param rho_base: Base keep ratio.
    :param w_structure: Initial weight for structure score (learnable).
    :param w_noise: Initial weight for noise penalty (learnable).
    :param w_empty: Initial weight for empty-patch penalty (learnable).
    :param tau: Temperature for score distribution entropy.
    """

    def __init__(
        self,
        rho_min: float = 0.15,
        rho_max: float = 0.35,
        rho_base: float = 0.25,
        w_structure: float = 0.25,
        w_noise: float = 0.20,
        w_empty: float = 0.50,
        tau: float = 0.07,
    ):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_base = rho_base
        self.w_structure = nn.Parameter(torch.tensor(float(w_structure)))
        self.w_noise = nn.Parameter(torch.tensor(float(w_noise)))
        self.w_empty = nn.Parameter(torch.tensor(float(w_empty)))
        self.tau = tau

        self.rho_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        word_tokens: torch.Tensor,
        event_cues: dict[str, torch.Tensor],
        hw: tuple,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Select tokens using event-aware scoring.

        :param visual_tokens: Patch tokens [B, N, C].
        :param word_tokens: Word tokens [B, L, C].
        :param event_cues: Dict from ``EventCueExtractor``.
        :param hw: Spatial grid size (H, W) with ``H * W == N``.
        :return: Tuple of
            ``selected_idx`` [B, K],
            ``word_idx`` [B, N],
            ``score`` [B, N],
            ``rho`` [B],
            ``score_map`` [B, 1, H, W].
        """
        b, n, c = visual_tokens.shape
        h, w = hw

        v = F.normalize(visual_tokens, dim=-1)
        t = F.normalize(word_tokens, dim=-1)
        sim = torch.matmul(v, t.transpose(1, 2))  # [B, N, L]
        semantic_score, word_idx = sim.max(dim=-1)  # [B, N]

        density = (
            event_cues["density"].flatten(2).transpose(1, 2).squeeze(-1)
        )  # [B, N]
        gradient = (
            event_cues["gradient"].flatten(2).transpose(1, 2).squeeze(-1)
        )  # [B, N]
        burst = (
            event_cues["burst"].flatten(2).transpose(1, 2).squeeze(-1)
        )  # [B, N]
        empty = (
            event_cues["empty_mask"].flatten(2).transpose(1, 2).squeeze(-1)
        )  # [B, N]

        structure_score = density * gradient
        noise_penalty = burst * (1.0 - gradient)

        score = (
            semantic_score
            + self.w_structure * structure_score
            - self.w_noise * noise_penalty
            - self.w_empty * (1.0 - empty)
        )

        # Text confidence: sharper distribution = more reliable semantics
        prob = F.softmax(score / self.tau, dim=1)
        entropy = -(prob * (prob + 1e-6).log()).sum(dim=1) / torch.log(
            torch.tensor(float(n), device=score.device)
        )
        score_mean = score.mean(dim=1)
        score_std = score.std(dim=1)
        density_mean = density.mean(dim=1)
        top_gap = score.topk(k=min(2, n), dim=1).values
        if top_gap.shape[1] >= 2:
            gap = top_gap[:, 0] - top_gap[:, 1]
        else:
            gap = torch.zeros_like(score_mean)

        stats = torch.stack(
            [score_mean, score_std, density_mean, 1.0 - entropy, gap], dim=1
        )
        delta = 0.10 * self.rho_mlp(stats).squeeze(1)
        rho = torch.clamp(
            self.rho_base + delta,
            min=self.rho_min,
            max=self.rho_max,
        )

        # Batch needs consistent K; use mean rho across batch
        k = max(1, int(round(float((rho.mean() * n).detach().cpu()))))
        selected_idx = torch.topk(score, k=k, dim=1).indices  # [B, K]
        score_map = score.view(b, 1, h, w)

        return selected_idx, word_idx, score, rho, score_map
