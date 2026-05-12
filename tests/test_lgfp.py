import torch

from src.models.components.mods.lgfp import LowFrequencyGuidedFeaturePurification
from src.models.components.network import SGVPRNet


def test_lgfp_module_shape():
    """LFP module should preserve spatial and channel dimensions."""
    x = torch.randn(2, 768, 14, 14)
    m = LowFrequencyGuidedFeaturePurification(in_channels=768)
    out = m(x)
    assert out.shape == x.shape


def test_lgfp_module_no_gaussian_smoothing():
    """LFP module should work when Gaussian smoothing is disabled."""
    x = torch.randn(2, 32, 16, 16)
    m = LowFrequencyGuidedFeaturePurification(
        in_channels=32, enable_gaussian_smoothing=False
    )
    out = m(x)
    assert out.shape == x.shape


def test_sgvpr_net_without_lgfp():
    """Default SGVPRNet should forward without LGFP."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_with_lgfp():
    """SGVPRNet with LGFP enabled should forward correctly."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
        mods={"lgfp": {"enabled": True}},
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_lgfp_with_frequency_enhance():
    """LGFP and frequency enhancement should coexist."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=True,
        mods={"lgfp": {"enabled": True}},
    )
    x = torch.randn(2, 3, 518, 518)
    out = net(x)
    assert out.shape == (2, 14 * 768)


def test_sgvpr_net_lgfp_with_other_mods():
    """LGFP should coexist with scfm and igr."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
        mods={
            "scfm": {"enabled": True},
            "igr": {"enabled": True},
            "lgfp": {"enabled": True},
        },
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)
