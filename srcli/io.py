"""Image loading and saving.

Kept separate from the pipeline because format handling carries its own set of
gotchas: colour space, bit depth, and how to treat the alpha channel.
"""

import numpy as np
import torch
from PIL import Image


def load_image(path):
  """Load an image as a float tensor in the 0..1 range.

  The tensor is always (1, 3, height, width). Every input is converted to three
  channel RGB because the networks are trained on colour and cannot consume an
  alpha channel, a palette index or a CMYK split. A transparent image is
  therefore flattened onto its own pixels, with the alpha discarded rather than
  silently mistaken for a colour channel.
  """
  image = Image.open(path).convert("RGB")
  array = np.asarray(image).astype(np.float32) / 255.0
  # PIL gives (H, W, C); torch wants (C, H, W), so move the last axis first.
  tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
  return tensor


def save_image(tensor, path):
  """Write a float tensor back to disk, choosing a sensible container.

  The tensor is expected to be (1, C, H, W) in the 0..1 range. Lossless output
  is preferred by default because re-encoding a freshly synthesised image to
  JPEG throws away exactly the detail we worked to add.
  """
  array = tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
  array = (array * 255.0 + 0.5).astype(np.uint8)
  # A single channel comes back as (H, W); PIL needs (H, W, C) or a 2D array,
  # and a 2D array is exactly what a greyscale image should be.
  if array.shape[2] == 1:
    array = array[:, :, 0]
  image = Image.fromarray(array)
  image.save(path)
