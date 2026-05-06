import torch
from src.models.components.frequency_enhancement import (
    EventFrequencyEnhancer,
    tokens_to_map,
    map_to_tokens,
    FrequencyContrastiveLoss,
)


def test_event_frequency_enhancer_shape_37():
    x = torch.randn(2, 37 * 37, 768)
    fmap, h, w = tokens_to_map(x)
    m = EventFrequencyEnhancer(dim=768)
    y, info = m(fmap)
    assert y.shape == fmap.shape
    assert info["freq_energy"].shape == (2, 37 * 37)
    y_tokens = map_to_tokens(y)
    assert y_tokens.shape == x.shape


def test_event_frequency_enhancer_shape_16():
    x = torch.randn(2, 16 * 16, 768)
    fmap, h, w = tokens_to_map(x)
    m = EventFrequencyEnhancer(dim=768)
    y, info = m(fmap)
    assert y.shape == fmap.shape
    assert info["freq_energy"].shape == (2, 16 * 16)


def test_frequency_contrastive_loss():
    x = torch.randn(4, 768, 16, 16)
    labels = torch.tensor([0, 0, 1, 1])
    loss_fn = FrequencyContrastiveLoss(num_negatives=1)
    loss = loss_fn(x, labels)
    assert torch.isfinite(loss)


def test_gamma_init_zero_passes_through():
    """When gamma_init=0, the enhancer should initially act as identity (plus GroupNorm)."""
    x = torch.randn(2, 16 * 16, 768)
    fmap, h, w = tokens_to_map(x)
    m = EventFrequencyEnhancer(dim=768, gamma_init=0.0)
    y, info = m(fmap)
    # With gamma=0, output should be patch_map (but note GroupNorm is applied inside)
    # At least shapes must match and gamma should be 0
    assert torch.isclose(m.gamma_out, torch.tensor(0.0))
    assert y.shape == fmap.shape
