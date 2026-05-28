import torch

from src.models.components.mods.sparse_prompt import SparsePromptModule
from src.models.components.network import SGVPRNet


def test_sparse_prompt_module_shape():
    """SPM module should preserve spatial and channel dimensions."""
    x = torch.randn(2, 768, 14, 14)
    m = SparsePromptModule(
        prompt_channels=768,
        num_prompt_experts=5,
        prompt_spatial_size=96,
        router_input_dim=768,
        num_active_experts=1,
    )
    out = m(x)
    assert out.shape == x.shape


def test_sgvpr_net_without_spm():
    """Default SGVPRNet should forward without SPM."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_with_spm():
    """SGVPRNet with SPM enabled should forward correctly."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=False,
        mods={"spm": {"enabled": True}},
    )
    x = torch.randn(1, 3, 518, 518)
    out = net(x)
    assert out.shape == (1, 14 * 768)


def test_sgvpr_net_spm_with_frequency_enhance():
    """SPM and frequency enhancement should coexist."""
    net = SGVPRNet(
        pretrained_foundation=False,
        foundation_model_path=None,
        use_clip_text=False,
        use_frequency_enhance=True,
        mods={"spm": {"enabled": True}},
    )
    x = torch.randn(2, 3, 518, 518)
    out = net(x)
    assert out.shape == (2, 14 * 768)
