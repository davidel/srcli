import pytest
import torch

from srcli.models import loader
from srcli.models.registry import ModelEntry
from srcli.models.sources import LocalFile


class TinyNet(torch.nn.Module):
  """A trivial network that scales by two, used to exercise the loader."""

  scale = 2

  def __init__(self):
    super().__init__()
    self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

  def forward(self, x):
    # Nearest neighbour doubling, so the measured factor is unambiguous.
    return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


def test_extract_state_descends_into_a_wrapper_key():
  inner = {"weight": torch.zeros(1)}
  wrapped = {"params_ema": inner}
  assert loader._extract_state(wrapped, "params_ema") == inner


def test_extract_state_passes_through_a_bare_state_dict():
  bare = {"weight": torch.zeros(1)}
  assert loader._extract_state(bare, "params_ema") == bare


def test_load_entry_loads_matching_weights(tmp_path):
  net = TinyNet()
  path = str(tmp_path / "tiny.pth")
  torch.save({"params_ema": net.state_dict()}, path)

  entry = ModelEntry(name="tiny",
                     scale=2,
                     build=TinyNet,
                     weights=LocalFile(path, state_key="params_ema"))
  loaded = loader.load_entry(entry, torch.device("cpu"))
  assert loaded.scale == 2
  assert loaded.alignment == 2


def test_load_entry_refuses_mismatched_weights(tmp_path):
  # A checkpoint for a different architecture must fail loudly rather than load
  # partially, which is the whole reason strict loading is used.
  path = str(tmp_path / "wrong.pth")
  torch.save({"params_ema": {"conv.weight": torch.zeros(3, 3, 3, 3)}}, path)

  entry = ModelEntry(name="tiny",
                     scale=2,
                     build=TinyNet,
                     weights=LocalFile(path, state_key="params_ema"))
  with pytest.raises(RuntimeError):
    loader.load_entry(entry, torch.device("cpu"))


def test_verify_scale_confirms_the_declared_factor():
  # Build the module directly, bypassing the weight fetch, since verify_scale
  # only cares about the forward pass.
  loaded = loader.LoadedModel(TinyNet(), scale=2, alignment=2, name="tiny")
  assert loader.verify_scale(loaded) == 2


def test_verify_scale_catches_a_wrong_declaration():
  loaded = loader.LoadedModel(TinyNet(), scale=4, alignment=2, name="tiny")
  with pytest.raises(ValueError):
    loader.verify_scale(loaded)
