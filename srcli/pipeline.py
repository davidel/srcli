"""Orchestration: turn an input image and a request into an output image.

This module ties the pieces together in a fixed order so the ordering bugs
cannot hide. The steps are load, resolve the plan, pad, infer (whole image or
tiled), crop away the halo, then resample to the exact requested size.
"""

import torch
from torch.nn import functional

from srcli import geometry
from srcli.device import pick_device, pick_dtype
from srcli.io import load_image, save_image
from srcli.models import loader, registry, select
from srcli.tiling import infer_tiled

# A rough ceiling on how many input pixels we push through the network at once.
# Exceeding it triggers tiling rather than a memory error. The value is a
# compromise: large enough that ordinary photos run in a single pass, small
# enough that a phone snapshot still fits on a modest GPU.
DEFAULT_TILE_PIXELS = 1024 * 1024


def _resolve_models(model, target_factor, tags):
  """Decide which models to run for this request.

  A named model is used as is; the caller has taken responsibility for it. The
  special name "auto" hands the decision to the selector, which may return a
  chain of several models when no single one reaches the requested factor.
  """
  if model != "auto":
    entry = registry.get(model)
    return [entry.name], entry.scale, None
  plan = select.select(target_factor, tags=tags)
  return plan["models"], plan["factor"], plan.get("warn")


def upscale(input_path,
            output_path,
            scale=None,
            target_size=None,
            model="auto",
            device=None,
            fp16=False,
            tile=None,
            tags=None):
  """Upscale a single image and write it to `output_path`.

  `scale` and `target_size` are mutually exclusive ways of describing the goal.
  Everything else is an optional override; sensible defaults are derived from the
  chosen model and the available hardware.
  """
  if scale is not None and target_size is not None:
    raise ValueError("provide either scale or target_size, not both")

  torch_device = pick_device(device)
  dtype = pick_dtype(torch_device, fp16)

  tensor = load_image(input_path).to(dtype=dtype)
  input_height, input_width = tensor.shape[2], tensor.shape[3]

  # Both ways of describing the goal collapse into an absolute target size in
  # output pixels, so the rest of the pipeline only deals with one form. The
  # factor is derived from it because model selection works in factors.
  if target_size is not None:
    target_width, target_height = target_size
    target_factor = max(target_width / input_width, target_height / input_height)
  elif scale is not None:
    target_factor = scale
    target_width = round(input_width * scale)
    target_height = round(input_height * scale)
  else:
    # Neither given: fall back to the model's own factor, which means no resize.
    target_factor = None
    target_width = None
    target_height = None

  if model == "auto" and target_factor is None:
    # Without a request there is nothing to select against, so pick the general
    # purpose default by name.
    model = "realesrgan-x4plus"
    target_factor = registry.get(model).scale

  names, factor, warn = _resolve_models(model, target_factor, tags)
  if warn:
    # A warning is not a failure; the run continues with a resample remainder.
    print(f"warning: {warn}")

  # The image is processed once per model in the chain. The factor used for the
  # geometry plan is the combined factor of the whole chain.
  loaded = [loader.load_by_name(name, torch_device, dtype=dtype) for name in names]
  alignment = 1
  for item in loaded:
    alignment = geometry.lcm(alignment, item.alignment)

  plan = geometry.plan_upscale(input_width,
                               input_height,
                               factor,
                               alignment,
                               target_width=target_width,
                               target_height=target_height)

  # Each link trims its own halo as it goes, so after the loop the result is
  # already the clean scaled image at the combined factor.
  result = tensor
  for item in loaded:
    result = _run_model(result, item, tile)

  # Finally, resample if the request was for a size the combined factor does not
  # hit exactly. Overshooting then shrinking looks better than a fractional
  # factor run directly.
  if plan["needs_resize"]:
    result = functional.interpolate(result,
                                    size=(plan["final_height"], plan["final_width"]),
                                    mode="bicubic",
                                    align_corners=False)

  save_image(result, output_path)
  return output_path


def _run_model(tensor, loaded, tile):
  """Run one model over the image, tiling if it would exceed the pixel budget.

  Padding to the model's alignment happens here rather than once up front,
  because a chain of models can have different alignment requirements and each
  link must satisfy its own.
  """
  input_height, input_width = tensor.shape[2], tensor.shape[3]
  pad = geometry.plan_reflect_padding(input_width, input_height, loaded.alignment)

  # Reflection is the desired padding, but it is only valid while the pad is
  # smaller than the dimension it mirrors. For a tiny image we fall back to
  # replication, which is defined for any pad size and still avoids the black
  # halo that zero padding would create.
  mode = "reflect" if pad["reflect_safe"] else "replicate"
  padded = functional.pad(tensor,
                          (pad["left"], pad["right"], pad["top"], pad["bottom"]),
                          mode=mode)

  budget = tile if tile is not None else DEFAULT_TILE_PIXELS
  padded_height, padded_width = padded.shape[2], padded.shape[3]
  if padded_height * padded_width <= budget:
    result = loaded.upscale(padded).to(torch.float32)
  else:
    # Choose a tile edge whose square is near the pixel budget, then let the
    # geometry code snap it to the alignment and lay out the grid. The edge is
    # kept at least two aligned steps wide so there is always room for a
    # strictly smaller overlap; otherwise a very small budget would ask for a
    # tile and an overlap of the same size, which has no valid layout.
    edge = geometry.snap_up(max(int(budget**0.5), 2 * loaded.alignment),
                            loaded.alignment)
    overlap = geometry.snap_up(max(loaded.alignment, edge // 8), loaded.alignment)
    # Guarantee the overlap stays below the tile, which plan_tiles requires.
    overlap = min(overlap, edge - loaded.alignment)
    tiles = geometry.plan_tiles(padded_width, padded_height, edge, overlap,
                                loaded.alignment)
    result = infer_tiled(padded, loaded, tiles, overlap, padded_width, padded_height)

  # Remove this link's halo so the next link in the chain sees a clean image.
  crop_left = pad["left"] * loaded.scale
  crop_top = pad["top"] * loaded.scale
  valid_width = input_width * loaded.scale
  valid_height = input_height * loaded.scale
  return result[:, :, crop_top:crop_top + valid_height,
                crop_left:crop_left + valid_width]
