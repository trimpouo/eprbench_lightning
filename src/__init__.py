"""Source package initialization."""

import functools

import torch

# PyTorch 2.6+ changed the default value of torch.load's weights_only argument
# from False to True. Existing checkpoints saved by this project (and by older
# Lightning versions) contain globals such as functools.partial and optimizer
# classes that are not in the default allowlist. Patch torch.load so that
# weights_only defaults to False for backwards compatibility.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# Additionally allowlist functools.partial for any explicit weights_only=True
# callers (e.g. user code or newer libraries).
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([functools.partial])
