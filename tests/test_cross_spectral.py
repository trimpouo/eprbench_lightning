import math

import torch

from src.models.components.mods.cross_spectral_refinement import (
    CrossSpectralAttentionFeatureRefinement,
)
from src.models.components.network import SGVPRNet


def test_cross_spectral_attention_module_shape():
    """CAFR module should produce high-level shaped output."""
    high = torch.randn(2, 768, 14, 14)
    low = torch.randn(2, 768, 28, 28)
    m = CrossSpectralAttentionFeatureRefinement(in_channels=768)
    out = m(high, low)
    assert out.shape == high.shape


def test_cross_spectral_attention_module_same_size():
    """CAFR should work when both inputs already share spatial size."""
    high = torch.randn(2, 32, 8, 8)
    low = torch.randn(2, 32, 8, 8)
    m = CrossSpectralAttentionFeatureRefinement(in_channels=32)
    out = m(high, low)
    assert out.shape == high.shape


def test_sgvpr_net_without_cross_spectral():
    """Default SGVPRNet should forward without cross-spectral refinement."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_with_cross_spectral():
    """SGVPRNet with cross-spectral enabled should forward correctly."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
        mods={"cross_spectral": {"enabled": True, "layer_idx": 6}},
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_cross_spectral_with_frequency_enhance():
    """Cross-spectral and frequency enhancement should coexist."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=True,
        mods={"cross_spectral": {"enabled": True, "layer_idx": 3}},
    )
    x = torch.randn(2, 3, 518, 518)
    out = net(x)
    assert out.shape == (2, 14 * 768)


def test_cross_spectral_hook_layer_idx_bounds():
    """Different valid layer indices should work."""
    for idx in [0, 5, 11]:
        net = SGVPRNet(
            pretrained_foundation=False,
            foundation_model_path=None,
            use_clip_text=False,
            use_frequency_enhance=False,
            mods={"cross_spectral": {"enabled": True, "layer_idx": idx}},
        )
        x = torch.randn(1, 3, 518, 518)
        out = net(x)
        assert out.shape == (1, 14 * 768)
