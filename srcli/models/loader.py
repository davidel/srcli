"""Turning a registry entry into a loaded, ready to run model.

This is the guardrail of the whole package. The architecture is built from the
entry, the weights are fetched from the entry's source, and the state dict is
loaded strictly. A mismatch between the two is a hard error rather than
something silently tolerated, because a half loaded network produces plausible
looking but wrong output.
"""

import torch

from srcli.models import registry


class LoadedModel:
  """A runnable module together with the facts the pipeline needs about it.

  It also plays the role the old Backend interface used to: the pipeline and the
  tiling code only ever need `scale`, `alignment` and a forward pass, so exposing
  `upscale` here removes a layer of indirection that no longer earned its keep.
  """

  def __init__(self, module, scale, alignment, name, device=None, dtype=None):
    self.module = module
    self.scale = scale
    self.alignment = alignment
    self.name = name
    self.device = device
    self.dtype = dtype

  def upscale(self, tensor):
    """Run a pre-padded image tensor through the network.

    The caller guarantees the shape already satisfies `alignment`, so this is a
    plain forward pass with no shape fixing. Gradient tracking is disabled
    because inference is the only thing we ever do here.
    """
    with torch.no_grad():
      return self.module(tensor.to(device=self.device, dtype=self.dtype))


def _extract_state(state, state_key):
  """Pull the tensor dictionary out of a checkpoint wrapper.

  Released checkpoints often nest the tensors under a key such as `params_ema`.
  When the key is present we descend into it; otherwise we assume the file is
  already a bare state dict.
  """
  if state_key and isinstance(state, dict) and state_key in state:
    return state[state_key]
  return state


def load_entry(entry, device, dtype=torch.float32):
  """Build the model described by `entry` and load its weights onto `device`."""
  path = entry.weights.fetch()
  # weights_only keeps the load strict and avoids executing anything embedded in
  # the file, which matters because checkpoints are frequently fetched remotely.
  state = torch.load(path, map_location="cpu", weights_only=True)
  state = _extract_state(state, getattr(entry.weights, "state_key", None))

  module = entry.build()
  # Strict loading is the check that the builder and the tensors agree. If they
  # do not, we want to know now rather than after a wasted inference pass.
  module.load_state_dict(state, strict=True)
  module.eval()
  module.to(device=device, dtype=dtype)
  return LoadedModel(module,
                     entry.scale,
                     entry.alignment,
                     entry.name,
                     device=device,
                     dtype=dtype)


def load_by_name(name, device, dtype=torch.float32):
  """Convenience wrapper that resolves a registry name before loading."""
  return load_entry(registry.get(name), device, dtype=dtype)


def verify_scale(loaded):
  """Confirm the model really scales by the factor the registry declares.

  A small probe image goes through the network once. The measured factor is
  compared against the declared one so a mislabelled entry is caught up front
  instead of after a full run. The probe uses the alignment as its size, which
  is the smallest input the network is known to accept.
  """
  size = max(1, loaded.alignment)
  probe = torch.zeros(1, 3, size, size)
  with torch.no_grad():
    out = loaded.module(probe)
  got = out.shape[-1] // size
  if got != loaded.scale:
    raise ValueError(f"registry declares x{loaded.scale} but the model produced x{got}")
  return got
