"""Tiling and seam blending for images that do not fit in memory.

The geometry module decides where the tiles go; this module moves the pixels
and hides the seams. The overlap between tiles is blended with a weighted
window so the network's slightly different output near each tile edge does not
show up as a grid.
"""

import torch


def _axis_weights(length, overlap, fade_start, fade_end):
  """Build a 1D blend profile that crossfades into its neighbours.

  The values are chosen so that a tile's trailing ramp plus the next tile's
  leading ramp sum to exactly one across the overlap. That makes the windows a
  partition of unity, which is what lets the tiles be combined without a seam.
  The ramp stops short of zero and one on purpose: hitting one would mean the
  neighbour contributes nothing at the shared pixel, so the two would sum to
  more than one.
  """
  weight = torch.ones(length)
  # Capping the ramp at half the length keeps the two ends from colliding on a
  # tile that is only a little larger than the overlap.
  ramp = min(overlap, length // 2)
  if ramp > 0:
    steps = torch.arange(1, ramp + 1, dtype=torch.float32) / (ramp + 1)
    if fade_start:
      weight[:ramp] = steps
    if fade_end:
      weight[length - ramp:] = torch.flip(steps, dims=[0])
  return weight


def _tile_weight(tile_width, tile_height, overlap, at_left, at_right, at_top, at_bottom):
  """Combine per-axis profiles into a 2D window for one tile.

  The window fades only on sides that meet another tile. A side that sits on
  the image boundary stays flat at one so the edge pixels are not dimmed.
  """
  wx = _axis_weights(tile_width, overlap, not at_left, not at_right)
  wy = _axis_weights(tile_height, overlap, not at_top, not at_bottom)
  return wy.unsqueeze(1) * wx.unsqueeze(0)


def infer_tiled(tensor, backend, tiles, overlap, image_width, image_height):
  """Run the backend over each tile and stitch the results back together.

  Tiles come from `geometry.plan_tiles`, so their coordinates are trusted. Each
  scaled tile is accumulated into a full size canvas together with a blend
  window, and the canvas is divided by the summed weights at the end. Dividing
  rather than plain averaging keeps the result correct even where the windows
  do not sum to one, such as in the corner shared by four tiles.
  """
  channels = tensor.shape[1]
  scale = backend.scale
  out_h = image_height * scale
  out_w = image_width * scale
  canvas = torch.zeros((channels, out_h, out_w), dtype=torch.float32)
  weights = torch.zeros((1, out_h, out_w), dtype=torch.float32)

  for x0, y0, x1, y1 in tiles:
    patch = tensor[:, :, y0:y1, x0:x1]
    scaled = backend.upscale(patch).to(torch.float32).squeeze(0)
    tile_h = (y1 - y0) * scale
    tile_w = (x1 - x0) * scale
    window = _tile_weight(tile_w, tile_h, overlap * scale, x0 == 0, x1 == image_width,
                          y0 == 0, y1 == image_height)
    ox, oy = x0 * scale, y0 * scale
    canvas[:, oy:oy + tile_h, ox:ox + tile_w] += scaled * window
    weights[:, oy:oy + tile_h, ox:ox + tile_w] += window

  # A pixel is always covered by at least one tile, so the weights are positive.
  return (canvas / weights).unsqueeze(0)
