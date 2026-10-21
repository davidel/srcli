"""Image loading and saving.

Kept separate from the pipeline because format handling carries its own set of
gotchas: colour space, bit depth, and how to treat the alpha channel.
"""

import numpy as np
import torch
from PIL import Image


def load_image(path):
  """Load an image as a float tensor in the 0..1 range.

  The tensor is shaped (1, channels, height, width) to match what the network
  expects. An alpha channel, if present, is kept rather than dropped so
  transparent regions survive the round trip; the caller decides what to do
  with it.
  """
  image = Image.open(path)
  # Anything that is not already RGB or RGBA (palette, greyscale, CMYK) is
  # converted, since the network is trained on three channel colour.
  if image.mode not in ("RGB", "RGBA"):
    image = image.convert("RGB")
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
