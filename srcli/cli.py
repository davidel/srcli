"""Command line entry point.

The CLI is intentionally thin: it only parses arguments, resolves the few
mutually exclusive options, and calls into the pipeline. Anything that looks like
logic belongs in a module that can be imported and tested without a terminal. We
use argparse rather than a decorator based parser so the tool stays free of type
annotations.
"""

import argparse
import sys

from srcli.models import registry
from srcli.pipeline import upscale


def build_parser():
  """Construct the argument parser."""
  parser = argparse.ArgumentParser(
      prog="sr",
      description="Upscale an image to a higher resolution using super-resolution.")
  parser.add_argument("input", help="path to the image to upscale")
  parser.add_argument("-o",
                      "--output",
                      required=True,
                      help="path to write the result to")

  # Scale and size are two ways of asking for the same thing, so they are kept
  # as separate flags and validated together rather than hidden behind a mode.
  parser.add_argument("--scale",
                      type=float,
                      help="upscale factor, e.g. 4 for four times larger")
  parser.add_argument("--size",
                      help="exact output size as WIDTHxHEIGHT, e.g. 3840x2160")

  parser.add_argument(
      "--model",
      default="auto",
      help="model name, or 'auto' to pick one for the requested factor; "
      "available: {}".format(", ".join(registry.names())))
  parser.add_argument(
      "--tag",
      action="append",
      dest="tags",
      help="prefer models with this tag when --model is auto, e.g. anime")
  parser.add_argument("--device", help="compute device: cpu, cuda or mps")
  parser.add_argument("--fp16",
                      action="store_true",
                      help="use half precision on CUDA to save memory and time")
  parser.add_argument(
      "--tile",
      type=int,
      help="input pixel budget before tiling kicks in; lower uses less memory")
  return parser


def parse_size(text):
  """Parse a WIDTHxHEIGHT string into a (width, height) pair."""
  try:
    width, height = text.lower().split("x")
    return int(width), int(height)
  except ValueError:
    raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT, e.g. 3840x2160")


def main(argv=None):
  """Parse arguments and dispatch to the pipeline."""
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.scale is not None and args.size is not None:
    parser.error("use either --scale or --size, not both")

  target_size = parse_size(args.size) if args.size is not None else None

  try:
    upscale(args.input,
            args.output,
            scale=args.scale,
            target_size=target_size,
            model=args.model,
            device=args.device,
            fp16=args.fp16,
            tile=args.tile,
            tags=args.tags)
  except (RuntimeError, ValueError, KeyError) as error:
    parser.error(str(error))
  return 0


if __name__ == "__main__":
  sys.exit(main())
