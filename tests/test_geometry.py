import pytest

from srcli import geometry


def test_pad_amount_reaches_next_multiple():
  assert geometry.pad_amount(10, 4) == 2
  assert geometry.pad_amount(12, 4) == 0
  assert geometry.pad_amount(13, 4) == 3
  # An alignment of one means no padding is ever needed.
  assert geometry.pad_amount(7, 1) == 0


def test_split_padding_biases_extra_pixel_to_trailing_edge():
  assert geometry.split_padding(2) == (1, 1)
  assert geometry.split_padding(3) == (1, 2)
  assert geometry.split_padding(0) == (0, 0)


def test_resolve_alignment_is_least_common_multiple():
  assert geometry.resolve_alignment(4) == 4
  assert geometry.resolve_alignment(4, 3) == 12
  assert geometry.resolve_alignment(1, 8) == 8


def test_reflect_padding_grows_to_alignment():
  plan = geometry.plan_reflect_padding(10, 6, 4)
  assert plan["padded_width"] == 12
  assert plan["padded_height"] == 8
  # Two columns and two rows of padding are needed in total.
  assert plan["left"] + plan["right"] == 2
  assert plan["top"] + plan["bottom"] == 2
  assert plan["reflect_safe"] is True


def test_reflect_padding_flags_unsafe_tiny_images():
  # A one pixel wide image cannot be reflected by even a single pixel.
  plan = geometry.plan_reflect_padding(1, 4, 4)
  assert plan["reflect_safe"] is False


def test_axis_starts_cover_the_axis_and_anchor_last_tile():
  starts = geometry.axis_starts(100, 40, 30)
  assert starts[0] == 0
  # The final tile must end exactly on the boundary.
  assert starts[-1] + 40 == 100
  # Every gap must be small enough that the overlap still applies.
  for a, b in zip(starts, starts[1:]):
    assert b - a <= 30


def test_axis_starts_single_tile_when_image_is_small():
  assert geometry.axis_starts(30, 40, 30) == [0]


def test_plan_tiles_snaps_to_alignment_and_raises_on_bad_overlap():
  tiles = geometry.plan_tiles(100, 60, 30, 10, 4)
  # 30 and 10 both snap up to 32 and 12 respectively.
  for x0, y0, x1, y1 in tiles:
    assert (x1 - x0) % 4 == 0
    assert (y1 - y0) % 4 == 0
  with pytest.raises(ValueError):
    geometry.plan_tiles(100, 60, 20, 20, 4)


def test_plan_tiles_covers_every_pixel():
  width, height = 100, 70
  tiles = geometry.plan_tiles(width, height, 32, 8, 4)
  covered = [[False] * width for _ in range(height)]
  for x0, y0, x1, y1 in tiles:
    for y in range(y0, y1):
      for x in range(x0, x1):
        covered[y][x] = True
  assert all(all(row) for row in covered)


def test_plan_upscale_natural_size_has_no_resize():
  plan = geometry.plan_upscale(10, 6, 4, 4)
  assert plan["valid_width"] == 40
  assert plan["valid_height"] == 24
  assert plan["needs_resize"] is False
  assert plan["final_width"] == 40


def test_plan_upscale_arbitrary_target_needs_resize():
  plan = geometry.plan_upscale(10, 6, 4, 4, target_width=50, target_height=50)
  assert plan["valid_width"] == 40
  assert plan["needs_resize"] is True
  assert plan["final_width"] == 50
  assert plan["final_height"] == 50


def test_plan_upscale_crop_box_matches_padding():
  plan = geometry.plan_upscale(10, 10, 2, 4)
  assert plan["crop_left"] == plan["padding"]["left"] * 2
  assert plan["crop_top"] == plan["padding"]["top"] * 2
