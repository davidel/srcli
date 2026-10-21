import torch

from srcli import geometry
from srcli.tiling import _axis_weights, infer_tiled


class FakeBackend:
  """A backend that doubles its input, so tiling can be tested without torch weights."""

  scale = 2
  alignment = 2

  def upscale(self, tensor):
    # Nearest neighbour doubling keeps the check exact and dependency free.
    return tensor.repeat_interleave(self.scale, dim=2).repeat_interleave(self.scale,
                                                                         dim=3)


def test_axis_weights_crossfade_sums_to_one():
  # The trailing ramp of one tile must exactly complement the leading ramp of
  # the next, otherwise a seam appears where they meet.
  leading = _axis_weights(20, 6, fade_start=True, fade_end=False)
  trailing = _axis_weights(20, 6, fade_start=False, fade_end=True)
  # The overlapping region is the first `6` of the right tile and the last `6`
  # of the left tile; read left to right they must add up to one everywhere.
  assert torch.allclose(leading[:6] + trailing[-6:], torch.ones(6))


def test_infer_tiled_reconstructs_a_constant_image():
  # A constant image makes seam blending observable: any windowing mistake shows
  # up as a ripple where tiles overlap.
  image = torch.full((1, 3, 40, 40), 0.5)
  tiles = geometry.plan_tiles(40, 40, 16, 4, 2)
  result = infer_tiled(image, FakeBackend(), tiles, 4, 40, 40)
  assert result.shape == (1, 3, 80, 80)
  assert torch.allclose(result, torch.full((1, 3, 80, 80), 0.5), atol=1e-5)


def test_infer_tiled_preserves_a_gradient_across_seams():
  # A linear ramp should come back exactly doubled. Because every tile sees the
  # same pixels, any deviation here would mean the blend windows do not sum to
  # one somewhere, which is precisely a visible seam.
  ramp = torch.linspace(0, 1, 40).view(1, 1, 1, 40).repeat(1, 1, 40, 1)
  tiles = geometry.plan_tiles(40, 40, 16, 4, 2)
  result = infer_tiled(ramp, FakeBackend(), tiles, 4, 40, 40)
  expected = ramp.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
  assert torch.allclose(result, expected, atol=1e-4)
