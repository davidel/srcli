"""Hardware selection.

Isolated so the rest of the code can ask a single question, "what device and
dtype should I use", instead of scattering torch.cuda checks everywhere.
"""

import torch


def pick_device(requested=None):
  """Return the torch device to use, honouring an explicit request.

  When nothing is requested we prefer a CUDA GPU, then Apple's MPS backend, and
  finally fall back to the CPU so the tool always runs. A requested device that
  is not actually available raises rather than silently downgrading, because
  picking the wrong accelerator can turn a quick run into a very slow one.
  """
  if requested is not None:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
      raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
      raise RuntimeError("MPS was requested but is not available")
    return device

  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def pick_dtype(device, fp16):
  """Return the tensor dtype for the given device and fp16 preference.

  Half precision is only a real win on CUDA. On the CPU it is either slower or
  unsupported, and on MPS support is uneven, so we quietly fall back to full
  precision there instead of failing the run.
  """
  if fp16 and device.type == "cuda":
    return torch.float16
  return torch.float32
