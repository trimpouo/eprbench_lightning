import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from src.models.components.backbone.vision_transformer import (
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)
from src.models.components.frequency_enhancement import (
    EventFrequencyEnhancer,
    FrequencyContrastiveLoss,
    map_to_tokens,
    tokens_to_map,
)
from src.models.components.mods.bim import BrainInspiredModule
from src.models.components.mods.cross_spectral_refinement import (
    CrossSpectralAttentionFeatureRefinement,
)
from src.models.components.mods.event_aware_token_selector import (
    EventAwareTokenSelector,
)
from src.models.components.mods.event_cues import EventCueExtractor
from src.models.components.mods.fbgw import FrequencyBandGuidedUnit
from src.models.components.mods.igr import InfoGatedReconstructUnit
from src.models.components.mods.lgfp import LowFrequencyGuidedFeaturePurification
from src.models.components.mods.sacf import (
    EventStabilitySACF,
    SemanticEventResidualSACF,
    SpectralSpatialAdaptiveFusion,
)
from src.models.components.mods.scfm import (
    EventReliableSCFM,
    SpatialChannelFeatureModulator,
)
from src.models.components.mods.sparse_prompt import SparsePromptModule
from src.models.components.mods.temporal_event_aggregator import (
    TemporalEventFrameAggregator,
)

# ImageNet normalization stats for approximate de-normalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        return (
            F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1)))
            .pow(1.0 / p)
            .unsqueeze(3)
        )
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class SGVPRNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(
        self,
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text: bool = True,
        clip_model_name: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b79k",
        clip_freeze: bool = True,
        use_frequency_enhance: bool = True,
        use_freq_topk: bool = True,
        freq_score_weight: float = 0.10,
        freq_low_cut: float = 0.25,
        freq_mid_cut: float = 0.60,
        freq_highpass_cut: float = 0.25,
        freq_max_gain: float = 0.50,
        freq_gamma_init: float = 0.0,
        mods: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.aggregation = nn.Sequential(
            L2Norm(), GeM(work_with_tokens=None), Flatten()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=16,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.1,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.text_alpha = nn.Parameter(torch.tensor(0.5))
        self.text_alpha_patch = nn.Parameter(torch.tensor(0.5))
        self.text_token_ln = nn.LayerNorm(768, eps=1e-6)
        self.mask_keep_ratio = 0.25
        if use_clip_text:
            from src.models.components.backbone.text_encoder_clip import CLIPTextEncoder

            self.text_encoder_clip = CLIPTextEncoder(
                model_name="ViT-B-16",
                pretrained="laion2b_s34b_b88k",
                output_dim=768,
                freeze_clip=True,
            )
        else:
            self.text_encoder_clip = None

        self.use_frequency_enhance = use_frequency_enhance
        self.use_freq_topk = use_freq_topk
        self.freq_score_weight = freq_score_weight
        self.embed_dim = 768

        self.mods = mods if mods is not None else {}
        sacf_cfg = self.mods.get("sacf", {})
        eats_cfg = self.mods.get("event_aware_token_selector", {})
        scfm_cfg = self.mods.get("scfm", {})

        # Event cues
        ec_cfg = self.mods.get("event_cues", {})
        sacf_variant = sacf_cfg.get("variant", None)
        if sacf_variant is None:
            if sacf_cfg.get("use_semantic_event_residual", False):
                sacf_variant = "se_rsacf"
            elif sacf_cfg.get("use_event_stability", False):
                sacf_variant = "event_stability"
            else:
                sacf_variant = "original"

        sacf_needs_event_cues = (
            sacf_cfg.get("enabled", False)
            and sacf_cfg.get("use_event_cues", False)
        ) or (
            sacf_cfg.get("enabled", False)
            and sacf_variant == "se_rsacf"
            and sacf_cfg.get("use_event_prior", True)
        )
        selector_needs_event_cues = eats_cfg.get("enabled", False)
        scfm_needs_event_cues = (
            scfm_cfg.get("enabled", False)
            and scfm_cfg.get("use_event_reliable", False)
            and scfm_cfg.get("use_event_cues", True)
        )
        self.use_event_cues = (
            ec_cfg.get("enabled", False)
            or sacf_needs_event_cues
            or selector_needs_event_cues
            or scfm_needs_event_cues
        )
        if self.use_event_cues:
            self.event_cue_extractor = EventCueExtractor(
                patch_size=ec_cfg.get("patch_size", 14)
            )

        # Event-aware token selector
        self.use_event_aware_selector = eats_cfg.get("enabled", False)
        if self.use_event_aware_selector:
            self.event_selector = EventAwareTokenSelector(
                rho_min=eats_cfg.get("rho_min", 0.15),
                rho_max=eats_cfg.get("rho_max", 0.35),
                rho_base=eats_cfg.get("rho_base", 0.25),
                w_structure=eats_cfg.get("w_structure", 0.25),
                w_noise=eats_cfg.get("w_noise", 0.20),
                w_empty=eats_cfg.get("w_empty", 0.50),
                tau=eats_cfg.get("tau", 0.07),
            )

        topk_cfg = self.mods.get("adaptive_text_topk", {})
        self.use_adaptive_text_topk = topk_cfg.get("enabled", False)
        self.adaptive_topk_rho_min = topk_cfg.get("rho_min", 0.20)
        self.adaptive_topk_rho_max = topk_cfg.get("rho_max", 0.30)
        self.adaptive_topk_rho_base = topk_cfg.get("rho_base", 0.25)
        self.adaptive_topk_entropy_weight = topk_cfg.get("entropy_weight", 0.05)
        self.adaptive_topk_tau = topk_cfg.get("tau", 0.07)

        # Temporal fusion
        tf_cfg = self.mods.get("temporal_fusion", {})
        self.use_temporal_fusion = tf_cfg.get("enabled", False)
        if self.use_temporal_fusion:
            self.temporal_aggregator = TemporalEventFrameAggregator(
                channels=self.embed_dim
            )

        # Cross-spectral
        cs_cfg = self.mods.get("cross_spectral", {})
        self.use_cross_spectral = cs_cfg.get("enabled", False)
        self.cross_spectral_layer_idx = cs_cfg.get("layer_idx", 6)
        if self.use_cross_spectral:
            self.cross_spectral_refiner = CrossSpectralAttentionFeatureRefinement(
                self.embed_dim
            )

        # SCFM / ER-SCFM
        self.use_scfm = scfm_cfg.get("enabled", False)
        self.use_event_reliable_scfm = scfm_cfg.get("use_event_reliable", False)
        if self.use_scfm:
            if self.use_event_reliable_scfm:
                self.scfm = EventReliableSCFM(
                    channels=self.embed_dim,
                    use_event_cues=(
                        self.use_event_cues
                        and scfm_cfg.get("use_event_cues", True)
                    ),
                )
            else:
                self.scfm = SpatialChannelFeatureModulator(
                    channels=self.embed_dim
                )

        # SACF / ES-SACF / SE-RSACF
        self.use_sacf = sacf_cfg.get("enabled", False)
        self.sacf_variant = sacf_variant
        self.use_semantic_event_sacf = self.sacf_variant == "se_rsacf"
        self.use_event_stability_sacf = (
            self.sacf_variant == "event_stability"
            or sacf_cfg.get("use_event_stability", False)
        )
        self.sacf_layer_idx = sacf_cfg.get("layer_idx", 6)
        layer_indices = sacf_cfg.get("layer_indices", None)
        if layer_indices is None:
            layer_indices = [self.sacf_layer_idx]
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        self.sacf_layer_indices = [int(idx) for idx in layer_indices]
        self.use_sacf_layer_mixture = sacf_cfg.get(
            "use_layer_mixture",
            len(self.sacf_layer_indices) > 1,
        )
        self.use_sacf_text_layer_gate = sacf_cfg.get("use_text_layer_gate", False)
        if self.use_sacf:
            self._validate_layer_indices(self.sacf_layer_indices, name="sacf")
            if self.use_sacf_layer_mixture and len(self.sacf_layer_indices) > 1:
                self.sacf_layer_logits = nn.Parameter(
                    torch.zeros(len(self.sacf_layer_indices))
                )
            else:
                self.sacf_layer_logits = None

            if (
                self.use_sacf_text_layer_gate
                and len(self.sacf_layer_indices) > 1
            ):
                self.sacf_text_layer_gate = nn.Linear(
                    self.embed_dim,
                    len(self.sacf_layer_indices),
                )
            else:
                self.sacf_text_layer_gate = None

            local_dilations = sacf_cfg.get("local_dilations", [1])
            if isinstance(local_dilations, int):
                local_dilations = [local_dilations]
            local_dilations = tuple(int(dilation) for dilation in local_dilations)

            if self.use_semantic_event_sacf:
                self.sacf = SemanticEventResidualSACF(
                    in_channels=self.embed_dim,
                    local_window_size=sacf_cfg.get("local_window_size", 3),
                    tau=sacf_cfg.get("tau", 0.1),
                    lambda_sem=sacf_cfg.get("lambda_sem", 0.3),
                    lambda_evt=sacf_cfg.get("lambda_evt", 0.2),
                    dilations=local_dilations,
                    gate_type=sacf_cfg.get("gate_type", "conv"),
                    gate_hidden=sacf_cfg.get("gate_hidden", 16),
                    gate_num_grids=sacf_cfg.get("gate_num_grids", 8),
                    gate_dropout=sacf_cfg.get("gate_dropout", 0.0),
                    residual_scale_init=sacf_cfg.get(
                        "residual_scale_init", 0.1
                    ),
                    zero_init_out_proj=sacf_cfg.get("zero_init_out_proj", True),
                    use_semantic_prior=sacf_cfg.get("use_semantic_prior", True),
                    use_event_prior=sacf_cfg.get("use_event_prior", True),
                    event_low_thr=sacf_cfg.get("event_low_thr", 0.03),
                    event_high_thr=sacf_cfg.get("event_high_thr", 0.95),
                    event_smooth=sacf_cfg.get("event_smooth", 0.10),
                )
            elif self.use_event_stability_sacf:
                self.sacf = EventStabilitySACF(
                    in_channels=self.embed_dim,
                    local_window_size=sacf_cfg.get("local_window_size", 3),
                    use_event_cues=(
                        self.use_event_cues
                        and sacf_cfg.get("use_event_cues", True)
                    ),
                    use_semantic_score=(
                        self.use_event_aware_selector
                        and sacf_cfg.get("use_semantic_score", True)
                    ),
                )
            else:
                self.sacf = SpectralSpatialAdaptiveFusion(
                    in_channels=self.embed_dim,
                    local_window_size=sacf_cfg.get("local_window_size", 3),
                )

        # Other optional mods (unchanged)
        igr_cfg = self.mods.get("igr", {})
        self.use_igr = igr_cfg.get("enabled", False)
        if self.use_igr:
            self.igr = InfoGatedReconstructUnit(
                in_channels=self.embed_dim,
                gate_threshold=igr_cfg.get("gate_threshold", 0.5),
                num_groups=igr_cfg.get("num_groups", 16),
            )

        lgfp_cfg = self.mods.get("lgfp", {})
        self.use_lgfp = lgfp_cfg.get("enabled", False)
        if self.use_lgfp:
            self.lgfp = LowFrequencyGuidedFeaturePurification(
                in_channels=self.embed_dim,
                wavelet_type=lgfp_cfg.get("wavelet_type", "haar"),
                padding_mode=lgfp_cfg.get("padding_mode", "symmetric"),
                enable_gaussian_smoothing=lgfp_cfg.get(
                    "enable_gaussian_smoothing", True
                ),
                high_frequency_threshold=lgfp_cfg.get(
                    "high_frequency_threshold", 0.5
                ),
            )

        fbgw_cfg = self.mods.get("fbgw", {})
        self.use_fbgw = fbgw_cfg.get("enabled", False)
        if self.use_fbgw:
            self.fbgw = FrequencyBandGuidedUnit(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                reduction_ratio=fbgw_cfg.get("reduction_ratio", 16),
            )

        bim_cfg = self.mods.get("bim", {})
        self.use_bim = bim_cfg.get("enabled", False)
        if self.use_bim:
            self.bim = BrainInspiredModule(channels=self.embed_dim)

        spm_cfg = self.mods.get("spm", {})
        self.use_spm = spm_cfg.get("enabled", False)
        if self.use_spm:
            self.spm = SparsePromptModule(
                prompt_channels=spm_cfg.get("prompt_channels", self.embed_dim),
                num_prompt_experts=spm_cfg.get("num_prompt_experts", 5),
                prompt_spatial_size=spm_cfg.get("prompt_spatial_size", 96),
                router_input_dim=spm_cfg.get("router_input_dim", self.embed_dim),
                num_active_experts=spm_cfg.get("num_active_experts", 1),
            )

        if self.use_frequency_enhance:
            self.freq_enhancer = EventFrequencyEnhancer(
                dim=self.embed_dim,
                low_cut=freq_low_cut,
                mid_cut=freq_mid_cut,
                highpass_cut=freq_highpass_cut,
                max_gain=freq_max_gain,
                gamma_init=freq_gamma_init,
            )
        else:
            self.freq_enhancer = None

    def _extract_backbone_features(self, x):
        """Helper to run backbone and return dict of features."""
        return self.backbone(x)

    def _get_event_cues(self, x_raw, hw):
        """Extract event cues if enabled, otherwise return None."""
        if not self.use_event_cues or x_raw is None:
            return None
        return self.event_cue_extractor(x_raw, out_hw=hw)

    def _validate_layer_indices(self, layer_indices, name: str) -> None:
        """Validate DINOv2 block indices used by optional cross-layer modules."""
        num_layers = len(self.backbone.blocks)
        invalid = [
            layer_idx
            for layer_idx in layer_indices
            if layer_idx < 0 or layer_idx >= num_layers
        ]
        if invalid:
            raise ValueError(
                f"{name} layer indices {invalid} are outside [0, {num_layers - 1}]."
            )

    def _hook_output_to_tokens(self, output):
        """Extract token tensor from a ViT block hook output."""
        if isinstance(output, tuple):
            return output[0]
        if isinstance(output, dict):
            return output.get("x", output.get("tokens"))
        return output

    def _tokens_to_patch_map(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dim: int,
    ) -> torch.Tensor:
        """Convert ViT tokens [B, 1+N, C] to a patch map [B, C, H, W]."""
        patch_tokens = tokens[:, 1:, :]
        return patch_tokens.view(batch_size, height, width, dim).permute(
            0, 3, 1, 2
        ).contiguous()

    def _mix_sacf_low_maps(
        self,
        layer_outputs: Dict[int, torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        dim: int,
        text: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Build the low-level SACF map from one or more hooked layers."""
        low_maps = []
        for layer_idx in self.sacf_layer_indices:
            tokens = layer_outputs.get(layer_idx)
            if tokens is None:
                continue
            low_maps.append(
                self._tokens_to_patch_map(tokens, batch_size, height, width, dim)
            )

        if len(low_maps) == 0:
            return None
        if len(low_maps) == 1 or not self.use_sacf_layer_mixture:
            return low_maps[0]

        if self.sacf_text_layer_gate is not None and text is not None:
            layer_weight = torch.softmax(self.sacf_text_layer_gate(text), dim=-1)
            mixed = torch.zeros_like(low_maps[0])
            for idx, low_map in enumerate(low_maps):
                mixed = mixed + layer_weight[:, idx].view(-1, 1, 1, 1) * low_map
            return mixed

        layer_weight = torch.softmax(self.sacf_layer_logits, dim=0)
        mixed = torch.zeros_like(low_maps[0])
        for weight, low_map in zip(layer_weight, low_maps):
            mixed = mixed + weight.view(1, 1, 1, 1) * low_map
        return mixed

    def _compute_semantic_score_map(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        hw: tuple[int, int],
    ) -> torch.Tensor | None:
        """Compute text-guided patch semantic scores from original high tokens."""
        if text_tokens is None:
            return None

        height, width = hw
        visual_tokens = F.normalize(visual_tokens, p=2, dim=-1)
        text_tokens = self.text_token_ln(text_tokens)
        text_tokens = F.normalize(text_tokens, p=2, dim=-1)
        sim_matrix = torch.bmm(visual_tokens, text_tokens.transpose(1, 2))
        if text_mask is not None:
            sim_matrix = sim_matrix.masked_fill(text_mask.unsqueeze(1) == 0, -1e4)

        max_sim = sim_matrix.max(dim=-1).values
        valid = max_sim > -1e3
        score = ((max_sim + 1.0) * 0.5).clamp(0.0, 1.0)
        score = torch.where(valid, score, torch.zeros_like(score))
        return score.view(score.shape[0], 1, height, width)

    def _adaptive_text_topk_count(
        self,
        semantic_score: torch.Tensor,
        num_tokens: int,
    ) -> tuple[int, torch.Tensor]:
        """Compute adaptive text top-k count from semantic score entropy."""
        prob = F.softmax(semantic_score / max(self.adaptive_topk_tau, 1e-6), dim=1)
        entropy = -(prob * (prob + 1e-6).log()).sum(dim=1) / torch.log(
            torch.tensor(float(num_tokens), device=semantic_score.device)
        )
        rho = torch.clamp(
            self.adaptive_topk_rho_base
            + self.adaptive_topk_entropy_weight * entropy,
            min=self.adaptive_topk_rho_min,
            max=self.adaptive_topk_rho_max,
        )
        k = max(1, int(round(float((rho.mean() * num_tokens).detach().cpu()))))
        return k, rho

    def forward(
        self,
        x,
        text: torch.Tensor = None,
        text_tokens: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        raw_texts: list = None,
        return_aux: bool = False,
        x_raw: torch.Tensor = None,
    ):
        """Forward pass with optional event-aware modules.

        :param x: Input images [B, C, H, W] or [B, T, C, H, W].
        :param text: Sentence-level text embeddings.
        :param text_tokens: Token-level text embeddings.
        :param text_mask: Attention mask for text tokens.
        :param raw_texts: List of raw text strings.
        :param return_aux: Whether to return auxiliary outputs.
        :param x_raw: Optional unnormalized raw images. If None and
            ``use_event_cues`` is enabled, an approximate de-normalization
            is applied internally.
        :return: Descriptors [B, D] or tuple with aux dict.
        """
        # Handle temporal input [B, T, C, H, W]
        if x.dim() == 5:
            B, T, C, H_img, W_img = x.shape
            x_flat = x.view(B * T, C, H_img, W_img)
            if x_raw is not None and x_raw.dim() == 5:
                x_raw_flat = x_raw.view(B * T, C, H_img, W_img)
            else:
                x_raw_flat = x_raw
            return self._forward_temporal(
                x_flat,
                text=text,
                text_tokens=text_tokens,
                text_mask=text_mask,
                raw_texts=raw_texts,
                return_aux=return_aux,
                x_raw=x_raw_flat,
                temporal_shape=(B, T),
            )

        return self._forward_single(
            x,
            text=text,
            text_tokens=text_tokens,
            text_mask=text_mask,
            raw_texts=raw_texts,
            return_aux=return_aux,
            x_raw=x_raw,
        )

    def _forward_single(
        self,
        x,
        text,
        text_tokens,
        text_mask,
        raw_texts,
        return_aux,
        x_raw,
    ):
        # Approximate raw if needed
        if x_raw is None and self.use_event_cues:
            device = x.device
            mean = _IMAGENET_MEAN.to(device)
            std = _IMAGENET_STD.to(device)
            x_raw = x * std + mean

        layer_outputs: Dict[int, torch.Tensor] = {}
        needs_low_level = self.use_cross_spectral or self.use_sacf
        hook_layer_indices = []
        if self.use_cross_spectral:
            hook_layer_indices.append(self.cross_spectral_layer_idx)
        if self.use_sacf:
            hook_layer_indices.extend(self.sacf_layer_indices)

        if needs_low_level:
            self._validate_layer_indices(hook_layer_indices, name="cross-layer")
            handles = []

            def _make_hook(layer_idx):
                def _hook(module, input, output):
                    layer_outputs[layer_idx] = self._hook_output_to_tokens(output)

                return _hook

            for layer_idx in sorted(set(hook_layer_indices)):
                handles.append(
                    self.backbone.blocks[layer_idx].register_forward_hook(
                        _make_hook(layer_idx)
                    )
                )
            x = self.backbone(x)
            for handle in handles:
                handle.remove()
        else:
            x = self.backbone(x)

        B, P, D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P - 1))
        x0 = x["x_norm_clstoken"]
        x_patch = x["x_prenorm"][:, 1:, :]

        # Text encoding is intentionally before SACF so semantic priors can
        # guide cross-layer fusion. Final top-k fusion still recomputes scores.
        if self.text_encoder_clip is not None and raw_texts is not None:
            text = self.text_encoder_clip(raw_texts)
            tokens_out = self.text_encoder_clip.encode_batch_tokens(raw_texts)
            text_tokens = tokens_out["tokens"].to(x0.device)
            text_mask = tokens_out["mask"].to(x0.device)

        # Event cues at token grid resolution
        event_cues = self._get_event_cues(x_raw, hw=(H, W))
        semantic_score_map = self._compute_semantic_score_map(
            x_patch,
            text_tokens,
            text_mask,
            hw=(H, W),
        )
        sacf_aux = None

        if needs_low_level and len(layer_outputs) > 0:
            orig_dtype = x_patch.dtype
            high_map = x_patch.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

            if self.use_cross_spectral:
                cross_tokens = layer_outputs.get(self.cross_spectral_layer_idx)
                if cross_tokens is not None:
                    cross_low_map = self._tokens_to_patch_map(
                        cross_tokens,
                        B,
                        H,
                        W,
                        D,
                    )
                    high_map = self.cross_spectral_refiner(high_map, cross_low_map)

            if self.use_sacf:
                low_map = self._mix_sacf_low_maps(
                    layer_outputs,
                    B,
                    H,
                    W,
                    D,
                    text=text,
                )
                if low_map is not None and self.use_semantic_event_sacf:
                    sacf_out = self.sacf(
                        high_map,
                        low_map,
                        event_cues=event_cues,
                        semantic_score_map=semantic_score_map,
                        return_aux=return_aux,
                    )
                    if return_aux:
                        high_map, sacf_aux = sacf_out
                    else:
                        high_map = sacf_out
                elif low_map is not None and self.use_event_stability_sacf:
                    high_map = self.sacf(
                        high_map,
                        low_map,
                        event_cues=event_cues,
                        semantic_score_map=semantic_score_map,
                    )
                elif low_map is not None:
                    high_map = self.sacf(high_map, low_map)

            high_map = high_map.to(orig_dtype)
            x_patch = high_map.permute(0, 2, 3, 1).reshape(B, W * H, D)

        image_global_raw = x0

        # Frequency enhancement
        freq_info: Dict[str, torch.Tensor] = {}
        patch_map = None
        if self.freq_enhancer is not None:
            patch_map, hp, wp = tokens_to_map(x_patch)
            patch_map, freq_info = self.freq_enhancer(patch_map)
            x_patch = map_to_tokens(patch_map)

        # Patch Fusion / Event-Aware Token Selection
        token_score_map = None
        topk_rho = None
        if text_tokens is not None:
            device = x_patch.device
            B, N, D = x_patch.shape

            t_tokens = self.text_token_ln(text_tokens)
            x_patch_norm = F.normalize(x_patch, p=2, dim=-1)
            t_tokens_norm = F.normalize(t_tokens, p=2, dim=-1)

            if self.use_event_aware_selector and event_cues is not None:
                topk_idx, word_idx, token_score, rho, score_map = (
                    self.event_selector(
                        x_patch_norm,
                        t_tokens_norm,
                        event_cues,
                        hw=(H, W),
                    )
                )
                # Update semantic score map for downstream SACF
                semantic_score_map = score_map
                topk_rho = rho

                selected_x = torch.gather(
                    x_patch, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D)
                )
                selected_token_idx = torch.gather(word_idx, 1, topk_idx)
                selected_text_feats = torch.gather(
                    t_tokens,
                    1,
                    selected_token_idx.unsqueeze(-1).expand(-1, -1, D),
                )
                # Adaptive injection strength per token
                token_score_selected = torch.gather(token_score, 1, topk_idx)
                alpha_i = torch.sigmoid(
                    self.text_alpha_patch + token_score_selected.unsqueeze(-1)
                )
                enhanced_part = selected_x * selected_text_feats
                x_out = x_patch.clone()
                x_out.scatter_add_(
                    1,
                    topk_idx.unsqueeze(-1).expand(-1, -1, D),
                    alpha_i * enhanced_part,
                )
                x_patch = x_out
                token_score_map = score_map
            else:
                # Original text-only top-k path
                sim_matrix = torch.bmm(
                    x_patch_norm, t_tokens_norm.transpose(1, 2)
                )
                if text_mask is not None:
                    sim_matrix = sim_matrix.masked_fill(
                        text_mask.unsqueeze(1) == 0, -1e4
                    )
                max_sim, max_token_idx = sim_matrix.max(dim=-1)

                score = max_sim
                if self.use_freq_topk and "freq_energy" in freq_info:
                    freq_energy = freq_info["freq_energy"].to(max_sim.dtype)
                    if freq_energy.shape == max_sim.shape:
                        score = score + self.freq_score_weight * freq_energy

                if self.use_adaptive_text_topk:
                    k, topk_rho = self._adaptive_text_topk_count(max_sim, N)
                else:
                    k = max(int(self.mask_keep_ratio * N), 1)
                _, topk_idx = torch.topk(
                    score, k=k, dim=1, largest=True, sorted=False
                )
                selected_x = torch.gather(
                    x_patch, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D)
                )
                selected_token_idx = torch.gather(max_token_idx, 1, topk_idx)
                selected_text_feats = torch.gather(
                    t_tokens,
                    1,
                    selected_token_idx.unsqueeze(-1).expand(-1, -1, D),
                )
                enhanced_part = selected_x * selected_text_feats
                x_out = x_patch.clone()
                x_out.scatter_add_(
                    1,
                    topk_idx.unsqueeze(-1).expand(-1, -1, D),
                    self.text_alpha_patch * enhanced_part,
                )
                x_patch = x_out

        # Reshape the latest patch tokens to a map. ``patch_map`` remains in
        # aux for optional frequency losses, but text fusion may have changed
        # ``x_patch`` after frequency enhancement.
        x_p = x_patch.view(B, H, W, D).permute(0, 3, 1, 2)

        if self.use_scfm:
            if self.use_event_reliable_scfm:
                x_p = self.scfm(x_p, event_cues=event_cues)
            else:
                x_p = self.scfm(x_p)

        if self.use_igr:
            x_p = self.igr(x_p)

        if self.use_lgfp:
            x_p = self.lgfp(x_p)

        if self.use_fbgw:
            x_p = self.fbgw(x_p)

        if self.use_bim:
            x_p = self.bim(x_p)

        if self.use_spm:
            spm_prompt = self.spm(x_p)
            x_p = x_p + spm_prompt

        x10, x11, x12, x13 = (
            self.aggregation(x_p[:, :, 0:8, 0:8]),
            self.aggregation(x_p[:, :, 0:8, 8:]),
            self.aggregation(x_p[:, :, 8:, 0:8]),
            self.aggregation(x_p[:, :, 8:, 8:]),
        )
        x20, x21, x22, x23, x24, x25, x26, x27, x28 = (
            self.aggregation(x_p[:, :, 0:5, 0:5]),
            self.aggregation(x_p[:, :, 0:5, 5:11]),
            self.aggregation(x_p[:, :, 0:5, 11:]),
            self.aggregation(x_p[:, :, 5:11, 0:5]),
            self.aggregation(x_p[:, :, 5:11, 5:11]),
            self.aggregation(x_p[:, :, 5:11, 11:]),
            self.aggregation(x_p[:, :, 11:, 0:5]),
            self.aggregation(x_p[:, :, 11:, 5:11]),
            self.aggregation(x_p[:, :, 11:, 11:]),
        )
        x = [
            i.unsqueeze(1)
            for i in [
                x0,
                x10,
                x11,
                x12,
                x13,
                x20,
                x21,
                x22,
                x23,
                x24,
                x25,
                x26,
                x27,
                x28,
            ]
        ]

        x = torch.cat(x, dim=1)
        x = self.encoder(x).view(B, 14 * D)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        if return_aux:
            aux = {
                "image_global_raw": image_global_raw,
                "patch_map": patch_map,
                "freq_info": freq_info,
                "event_cues": event_cues,
                "token_score_map": token_score_map,
                "semantic_score_map": semantic_score_map,
                "topk_rho": topk_rho,
                "sacf_aux": sacf_aux,
            }
            return x, aux
        return x

    def _forward_temporal(
        self,
        x_flat,
        text,
        text_tokens,
        text_mask,
        raw_texts,
        return_aux,
        x_raw,
        temporal_shape,
    ):
        B, T = temporal_shape
        # Run single-frame logic for all frames
        out_list = []
        aux_list = []
        for t_i in range(T):
            x_t = x_flat[t_i * B : (t_i + 1) * B]
            x_raw_t = None
            if x_raw is not None:
                x_raw_t = x_raw[t_i * B : (t_i + 1) * B]
            out_t, aux_t = self._forward_single(
                x_t,
                text=text,
                text_tokens=text_tokens,
                text_mask=text_mask,
                raw_texts=raw_texts,
                return_aux=True,
                x_raw=x_raw_t,
            )
            out_list.append(out_t)
            aux_list.append(aux_t)

        desc_t = torch.stack(out_list, dim=1)  # [B, T, D]

        # Simple mean fusion for descriptors (can be upgraded to TEFA)
        desc = desc_t.mean(dim=1)
        desc = F.normalize(desc, p=2, dim=-1)

        if return_aux:
            aux = {
                "image_global_raw": aux_list[0]["image_global_raw"],
                "patch_map": aux_list[0]["patch_map"],
                "freq_info": aux_list[0]["freq_info"],
                "event_cues": aux_list[0].get("event_cues"),
                "token_score_map": aux_list[0].get("token_score_map"),
                "semantic_score_map": aux_list[0].get("semantic_score_map"),
                "topk_rho": aux_list[0].get("topk_rho"),
                "sacf_aux": aux_list[0].get("sacf_aux"),
                "frame_descriptors": desc_t,
            }
            return desc, aux
        return desc


def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, (
            "Please specify foundation model path."
        )
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone
