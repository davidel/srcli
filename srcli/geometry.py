"""Pure geometry for the super-resolution pipeline.

Everything in here is plain arithmetic with no tensors or file access, which
makes the awkward parts of the pipeline (padding, tiling and cropping) easy to
test in isolation. The rest of the code trusts these helpers so that the
off-by-one mistakes stay in one place.
"""

import math


def lcm(a, b):
  """Least common multiple of two positive integers."""
  return a // math.gcd(a, b) * b


def resolve_alignment(network_multiple, extra_multiple=1):
  """Combine two divisibility requirements into one alignment value.

  A super-resolution network often only accepts inputs whose width and height
  are multiples of some small number (its internal down/up sampling factor).
  The tiling code has its own grid requirement. Both have to hold at the same
  time, so the alignment we settle on is the least common multiple.
  """
  return lcm(max(1, network_multiple), max(1, extra_multiple))


def snap_up(value, alignment):
  """Round `value` up to the next multiple of `alignment`."""
  if alignment <= 1:
    return value
  return (value + alignment - 1) // alignment * alignment


def pad_amount(size, alignment):
  """Number of pixels to add to `size` to reach a multiple of `alignment`."""
  if alignment <= 1:
    return 0
  return (-size) % alignment


def split_padding(total):
  """Split a total padding amount into (before, after).

  The odd pixel is biased to the trailing edge. This keeps the valid region
  anchored at the top-left corner, which is what the crop step assumes.
  """
  before = total // 2
  after = total - before
  return before, after


def plan_reflect_padding(width, height, alignment):
  """Work out how to grow an image to a multiple of `alignment`.

  We pad by reflection rather than with zeros. Zero padding creates a hard
  black edge that the network happily sharpens into a visible halo, whereas a
  mirrored edge looks like plausible image content. Reflection padding in
  PyTorch requires the pad to be smaller than the dimension it mirrors, so the
  result reports whether the plan is safe for a strict reflect mode.
  """
  pad_w = pad_amount(width, alignment)
  pad_h = pad_amount(height, alignment)
  left, right = split_padding(pad_w)
  top, bottom = split_padding(pad_h)
  return {
      "left": left,
      "right": right,
      "top": top,
      "bottom": bottom,
      "padded_width": width + pad_w,
      "padded_height": height + pad_h,
      # A single reflection cannot mirror more pixels than the dimension itself.
      "reflect_safe": pad_w < width and pad_h < height,
  }


def axis_starts(size, tile_size, step):
  """Starting offsets of tiles along one axis.

  The final tile is anchored to the trailing edge so it ends exactly on the
  image boundary. Without that anchor the last strip would be short and leave
  a thin uncovered band once blending trims the overlaps.
  """
  if size <= tile_size:
    return [0]
  starts = []
  pos = 0
  while pos + tile_size < size:
    starts.append(pos)
    pos += step
  last = size - tile_size
  if starts[-1] != last:
    starts.append(last)
  return starts


def plan_tiles(width, height, tile_size, overlap, alignment):
  """Lay out an overlapping grid of tiles covering the whole image.

  Tiles are returned as (x0, y0, x1, y1) with the lower bound inclusive and the
  upper bound exclusive. Both the tile size and the overlap are snapped up to
  the alignment so every tile handed to the network is a valid shape, and the
  step between tiles is therefore aligned as well. The overlap gives the
  blending step enough shared context to hide seams.
  """
  tile_size = snap_up(tile_size, alignment)
  overlap = snap_up(overlap, alignment)
  if overlap >= tile_size:
    raise ValueError("overlap must be smaller than the tile size")
  step = tile_size - overlap
  xs = axis_starts(width, tile_size, step)
  ys = axis_starts(height, tile_size, step)
  tiles = []
  for y0 in ys:
    for x0 in xs:
      tiles.append((x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)))
  return tiles


def plan_upscale(input_width,
                 input_height,
                 model_scale,
                 alignment,
                 target_width=None,
                 target_height=None):
  """Turn a user request into the exact operations the pipeline must perform.

  The caller supplies the original size, the integer factor the chosen model
  provides, the alignment, and an optional requested output size. We return the
  padding plan, the size the network will actually produce, the crop box that
  removes the reflection halo, and the final resize needed to hit the requested
  size.

  Because we can only run the model at its own integer factor, a request for an
  arbitrary size is met by running the model first and then resampling down (or
  up) to the target. Overshooting and shrinking looks far better than trying to
  synthesise a fractional factor directly.
  """
  pad = plan_reflect_padding(input_width, input_height, alignment)

  # The network output covers the padded input; the valid region is the part
  # that corresponds to real pixels rather than the reflected halo.
  scaled_width = pad["padded_width"] * model_scale
  scaled_height = pad["padded_height"] * model_scale
  valid_width = input_width * model_scale
  valid_height = input_height * model_scale

  if target_width is None or target_height is None:
    final_width, final_height = valid_width, valid_height
  else:
    final_width, final_height = target_width, target_height

  return {
      "padding": pad,
      "scaled_width": scaled_width,
      "scaled_height": scaled_height,
      "crop_left": pad["left"] * model_scale,
      "crop_top": pad["top"] * model_scale,
      "valid_width": valid_width,
      "valid_height": valid_height,
      "final_width": final_width,
      "final_height": final_height,
      "needs_resize": (final_width, final_height) != (valid_width, valid_height),
  }
