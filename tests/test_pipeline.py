"""End to end checks of the pipeline using a locally registered fake model.

These tests drive the real orchestration (pad, infer, crop, resample, tile) with
a stand-in network, so they run instantly and need no weights. They are the
place where size mistakes such as a double crop or a missing resize show up.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from srcli import pipeline
from srcli.models import registry
from srcli.models.registry import ModelEntry
from srcli.models.sources import LocalFile


class _Doubler(torch.nn.Module):
  """A network that doubles both spatial dimensions."""

  def forward(self, x):
    return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
  """Register a local scale-2 model and return its registry name."""
  path = str(tmp_path / "doubler.pth")
  torch.save({"params_ema": _Doubler().state_dict()}, path)
  entry = ModelEntry(name="fake-x2",
                     scale=2,
                     build=_Doubler,
                     weights=LocalFile(path, state_key="params_ema"))
  # Patch the shared registry for the duration of the test only.
  monkeypatch.setitem(registry.REGISTRY, "fake-x2", entry)
  return "fake-x2"


@pytest.fixture
def sample_image(tmp_path):
  """An odd sized image so padding and cropping are actually exercised."""
  path = str(tmp_path / "in.png")
  array = (np.random.rand(37, 53, 3) * 255).astype(np.uint8)
  Image.fromarray(array).save(path)
  return path


def test_upscale_exact_factor_matches_input_times_scale(fake_model, sample_image,
                                                        tmp_path):
  out = str(tmp_path / "out.png")
  pipeline.upscale(sample_image, out, scale=2, model=fake_model, device="cpu")
  assert Image.open(out).size == (106, 74)


def test_upscale_arbitrary_target_resizes_to_exact_size(fake_model, sample_image,
                                                        tmp_path):
  out = str(tmp_path / "out.png")
  pipeline.upscale(sample_image,
                   out,
                   target_size=(100, 100),
                   model=fake_model,
                   device="cpu")
  assert Image.open(out).size == (100, 100)


def test_upscale_tiling_path_matches_untiled_size(fake_model, sample_image, tmp_path):
  # Forcing a tiny tile budget must not change the output size; it only changes
  # how the work is split up.
  out = str(tmp_path / "out.png")
  pipeline.upscale(sample_image, out, scale=2, model=fake_model, device="cpu", tile=64)
  assert Image.open(out).size == (106, 74)


def test_upscale_rejects_scale_and_target_together(fake_model, sample_image, tmp_path):
  out = str(tmp_path / "out.png")
  with pytest.raises(ValueError):
    pipeline.upscale(sample_image,
                     out,
                     scale=2,
                     target_size=(100, 100),
                     model=fake_model)
