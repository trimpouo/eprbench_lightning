import torch

from src.models.components.mods.sacf import (
    CosineSpectralFeatureAggregator,
    FastKANGate,
    SemanticEventResidualSACF,
)
from src.models.components.network import SGVPRNet


def test_cosine_spectral_feature_aggregator_shape():
    """Cosine spectral aggregation should preserve high-level map shape."""
    high = torch.randn(2, 32, 8, 8)
    semantic = torch.rand(2, 1, 8, 8)
    event = torch.rand(2, 1, 8, 8)
    module = CosineSpectralFeatureAggregator(
        in_channels=32,
        local_window_size=3,
        dilations=(1, 2),
    )

    out = module(high, high, semantic_score_map=semantic, event_reliability_map=event)

    assert out.shape == high.shape


def test_fastkan_gate_shape():
    """FastKAN gate should produce branch logits at every patch location."""
    x = torch.randn(2, 4, 8, 8)
    gate = FastKANGate(in_dim=4, hidden=8, out_dim=3, num_grids=5)

    out = gate(x)

    assert out.shape == (2, 3, 8, 8)


def test_semantic_event_residual_sacf_aux_zero_start():
    """SE-RSACF should preserve shape and return auxiliary regularizers."""
    high = torch.randn(2, 32, 8, 8)
    low = torch.randn(2, 32, 8, 8)
    semantic = torch.rand(2, 1, 8, 8)
    event_cues = {
        "density": torch.rand(2, 1, 8, 8),
        "gradient": torch.rand(2, 1, 8, 8),
        "burst": torch.rand(2, 1, 8, 8),
        "empty_mask": torch.ones(2, 1, 8, 8),
    }
    module = SemanticEventResidualSACF(
        in_channels=32,
        gate_type="conv",
        residual_scale_init=0.1,
        zero_init_out_proj=True,
    )

    out, aux = module(
        high,
        low,
        event_cues=event_cues,
        semantic_score_map=semantic,
        return_aux=True,
    )

    assert out.shape == high.shape
    assert torch.allclose(out, high)
    assert aux["gate_low"].shape == (2, 1, 8, 8)
    assert aux["residual_loss"].ndim == 0
    assert aux["noise_loss"].ndim == 0


def test_semantic_event_residual_sacf_fastkan_gate_shape():
    """SE-RSACF should support the FastKAN gate option."""
    high = torch.randn(2, 32, 8, 8)
    low = torch.randn(2, 32, 8, 8)
    module = SemanticEventResidualSACF(
        in_channels=32,
        gate_type="fastkan",
        gate_hidden=8,
        gate_num_grids=5,
        zero_init_out_proj=False,
    )

    out = module(high, low)

    assert out.shape == high.shape


def test_sgvpr_net_with_se_rsacf_multilayer_and_adaptive_topk():
    """SGVPRNet should run with SE-RSACF, multi-layer mixing, and text top-k."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
        mods={
            "sacf": {
                "enabled": True,
                "variant": "se_rsacf",
                "layer_indices": [2, 4],
                "use_layer_mixture": True,
                "use_text_layer_gate": True,
                "local_dilations": [1],
                "gate_type": "fastkan",
                "gate_hidden": 8,
                "gate_num_grids": 5,
            },
            "event_cues": {"enabled": True, "patch_size": 14},
            "adaptive_text_topk": {"enabled": True},
        },
    )
    net.eval()
    x = torch.randn(1, 3, 518, 518)
    text_tokens = torch.randn(1, 6, 768)
    text_mask = torch.ones(1, 6, dtype=torch.long)

    with torch.no_grad():
        out, aux = net(
            x,
            text_tokens=text_tokens,
            text_mask=text_mask,
            return_aux=True,
        )

    assert out.shape == (1, 14 * 768)
    assert aux["sacf_aux"] is not None
    assert aux["semantic_score_map"].shape == (1, 1, 37, 37)
    assert aux["topk_rho"] is not None
    assert 0.20 <= float(aux["topk_rho"].mean()) <= 0.30
