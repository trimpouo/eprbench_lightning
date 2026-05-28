"""Sparse Prompt Module (SPM).

Reference:
    Beyond Degradation Redundancy: Contrastive Prompt Learning for All-in-One
    Image Restoration (TPAMI 2026)
    https://arxiv.org/pdf/2504.09973

The module dynamically selects the most suitable task prompt via sparse
routing to reduce redundancy and boost representation quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePromptModule(nn.Module):
    """Sparse prompt generation module with a learnable expert bank and router."""

    def __init__(
        self,
        prompt_channels: int = 128,
        num_prompt_experts: int = 5,
        prompt_spatial_size: int = 96,
        router_input_dim: int = 192,
        num_active_experts: int = 1,
    ):
        super().__init__()
        self.prompt_expert_bank = nn.Parameter(
            torch.rand(
                1,
                num_prompt_experts,
                prompt_channels,
                prompt_spatial_size,
                prompt_spatial_size,
            ),
            requires_grad=True,
        )
        self.prompt_router = nn.Linear(router_input_dim, num_prompt_experts)
        self.prompt_refinement_conv = nn.Conv2d(
            prompt_channels,
            prompt_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.num_active_experts = num_active_experts
        self.num_prompt_experts = num_prompt_experts

    def forward(
        self,
        input_features: torch.Tensor,
        generate_negative_prompt: bool = False,
    ) -> torch.Tensor:
        batch_size, channels, height, width = input_features.shape
        global_feature_descriptor = input_features.mean(dim=(-2, -1))
        router_logits = self.prompt_router(global_feature_descriptor)
        routing_probabilities = F.softmax(router_logits, dim=1).to(
            input_features.dtype
        )

        _, active_expert_indices = torch.topk(
            routing_probabilities,
            self.num_active_experts,
            dim=1,
        )

        if not generate_negative_prompt:
            sparse_routing_weights = torch.zeros_like(routing_probabilities)
            active_expert_weights = routing_probabilities.gather(
                1, active_expert_indices
            )
            sparse_routing_weights.scatter_(
                dim=1,
                index=active_expert_indices,
                src=active_expert_weights,
            )
            generated_prompt = self._compose_prompt_from_experts(
                sparse_routing_weights, batch_size
            )
        else:
            with torch.no_grad():
                negative_candidate_mask = torch.ones_like(
                    routing_probabilities, dtype=torch.bool
                )
                negative_candidate_mask.scatter_(
                    dim=1,
                    index=active_expert_indices,
                    value=False,
                )
                random_sampling_scores = torch.rand_like(routing_probabilities)
                random_sampling_scores.masked_fill_(
                    ~negative_candidate_mask, -float("inf")
                )
                _, negative_expert_indices = torch.topk(
                    random_sampling_scores,
                    self.num_active_experts,
                    dim=1,
                )
                negative_expert_weights = routing_probabilities.gather(
                    1, negative_expert_indices
                )
                sparse_routing_weights = torch.zeros_like(routing_probabilities)
                sparse_routing_weights.scatter_(
                    dim=1,
                    index=negative_expert_indices,
                    src=negative_expert_weights,
                )
                generated_prompt = self._compose_prompt_from_experts(
                    sparse_routing_weights, batch_size
                ).detach()

        generated_prompt = F.interpolate(
            generated_prompt,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        generated_prompt = self.prompt_refinement_conv(generated_prompt)
        return generated_prompt

    def _compose_prompt_from_experts(
        self, sparse_routing_weights: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        expanded_routing_weights = (
            sparse_routing_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        batched_prompt_experts = self.prompt_expert_bank.expand(
            batch_size, -1, -1, -1, -1
        )
        weighted_prompt_experts = (
            expanded_routing_weights * batched_prompt_experts
        )
        combined_prompt = torch.sum(weighted_prompt_experts, dim=1)
        return combined_prompt
