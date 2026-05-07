import torch

from src.models.components.mods.scfm import SpatialChannelFeatureModulator
from src.models.components.network import SGVPRNet


def test_spatial_channel_feature_modulator_shape():
    """SCFM module should preserve spatial and channel dimensions."""
    x = torch.randn(2, 768, 14, 14)
    m = SpatialChannelFeatureModulator(channels=768)
    out = m(x)
    assert out.shape == x.shape


def test_sgvpr_net_without_scfm():
    """Default SGVPRNet should forward without SCFM."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_with_scfm():
    """SGVPRNet with SCFM enabled should forward correctly."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
        mods={"scfm": {"enabled": True}},
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_scfm_with_frequency_enhance():
    """SCFM and frequency enhancement should coexist."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=True,
        mods={"scfm": {"enabled": True}},
    )
    x = torch.randn(2, 3, 518, 518)
    out = net(x)
    assert out.shape == (2, 14 * 768)
