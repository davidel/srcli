"""Tests for image loading and saving.

These cover the formats a user is likely to hand us and, most importantly, the
channel handling. A network trained on three channel colour cannot accept an
alpha channel, so the loader must always produce exactly three channels however
the source image was stored.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from srcli.io import load_image, save_image


def _write(path, array, mode=None):
  Image.fromarray(array, mode).save(path)


def test_load_rgb_image_gives_three_channels(tmp_path):
  path = str(tmp_path / "rgb.png")
  _write(path, (np.random.rand(40, 60, 3) * 255).astype(np.uint8))
  tensor = load_image(path)
  assert tuple(tensor.shape) == (1, 3, 40, 60)
  assert float(tensor.min()) >= 0.0
  assert float(tensor.max()) <= 1.0


def test_load_rgba_image_drops_alpha(tmp_path):
  # A transparent image must be flattened to RGB, not passed through as four
  # channels, which the network would reject.
  path = str(tmp_path / "rgba.png")
  _write(path, (np.random.rand(40, 60, 4) * 255).astype(np.uint8), "RGBA")
  assert tuple(load_image(path).shape) == (1, 3, 40, 60)


def test_load_grayscale_image_expands_to_three_channels(tmp_path):
  path = str(tmp_path / "grey.png")
  _write(path, (np.random.rand(40, 60) * 255).astype(np.uint8), "L")
  assert tuple(load_image(path).shape) == (1, 3, 40, 60)


def test_load_jpeg_image(tmp_path):
  path = str(tmp_path / "photo.jpg")
  _write(path, (np.random.rand(40, 60, 3) * 255).astype(np.uint8))
  assert tuple(load_image(path).shape) == (1, 3, 40, 60)


def test_save_round_trips_an_rgb_image(tmp_path):
  path = str(tmp_path / "out.png")
  tensor = torch.rand(1, 3, 20, 30)
  save_image(tensor, path)
  reloaded = load_image(path)
  assert tuple(reloaded.shape) == (1, 3, 20, 30)
  # Values are quantised to 8 bits on the way out, so allow one step of slack.
  assert torch.allclose(reloaded, tensor, atol=1.0 / 255.0)


def test_save_writes_a_jpeg_when_asked(tmp_path):
  path = str(tmp_path / "out.jpg")
  save_image(torch.rand(1, 3, 20, 30), path)
  assert Image.open(path).size == (30, 20)


def test_load_missing_file_raises(tmp_path):
  with pytest.raises(FileNotFoundError):
    load_image(str(tmp_path / "nope.png"))
