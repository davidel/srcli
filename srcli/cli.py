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

# The subcommands the parser knows about. Any invocation that does not start with
# one of these is treated as the default upscale command, so the short form
# `sr photo.png -o out.png` keeps working without naming the command.
SUBCOMMANDS = ("upscale", "models")

# Short labels for the weight source classes, used when listing the catalog.
SOURCE_LABELS = {
    "HubFile": "hf",
    "UrlFile": "url",
    "GDriveFile": "gdrive",
    "LocalFile": "local",
}


def build_parser():
  """Construct the argument parser with its subcommands."""
  parser = argparse.ArgumentParser(
      prog="sr",
      description="Upscale an image to a higher resolution using super-resolution.")
  subparsers = parser.add_subparsers(dest="command")

  _build_upscale_parser(subparsers)
  _build_models_parser(subparsers)
  return parser


def _build_upscale_parser(subparsers):
  """Register the upscale command and its options."""
  parser = subparsers.add_parser("upscale",
                                 help="upscale an image (the default command)")
  parser.add_argument("input", help="path to the image to upscale")
  parser.add_argument("-o",
                      "--output",
                      required=True,
                      help="path to write the result to")

  # Scale and size are two ways of asking for the same thing, so they are kept
  # as separate flags and validated together rather than hidden behind a mode.
  # The size is parsed here rather than later so a malformed value is reported
  # by argparse with the usual usage message.
  parser.add_argument("--scale",
                      type=float,
                      help="upscale factor, e.g. 4 for four times larger")
  parser.add_argument("--size",
                      type=parse_size,
                      help="exact output size as WIDTHxHEIGHT, e.g. 3840x2160")

  parser.add_argument(
      "--model",
      default="auto",
      help="model name, or 'auto' to pick one for the requested factor; "
      "available: {}".format(", ".join(registry.names())))
  parser.add_argument("--tag",
                      action="append",
                      dest="tags",
                      help="prefer models with this tag when --model is auto")
  parser.add_argument("--device", help="compute device: cpu, cuda or mps")
  parser.add_argument("--fp16",
                      action="store_true",
                      help="use half precision on CUDA to save memory and time")
  parser.add_argument(
      "--tile",
      type=int,
      help="input pixel budget before tiling kicks in; lower uses less memory")
  return parser


def _build_models_parser(subparsers):
  """Register the models command, which only lists the catalog."""
  return subparsers.add_parser("models", help="list the available models")


def parse_size(text):
  """Parse a WIDTHxHEIGHT string into a (width, height) pair."""
  try:
    width, height = text.lower().split("x")
    return int(width), int(height)
  except ValueError:
    raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT, e.g. 3840x2160")


def list_models(stream=None):
  """Print the registered models as an aligned table.

  The columns are the facts needed to choose a model: what factor it provides,
  what it is suited to, and where its weights come from.
  """
  stream = stream if stream is not None else sys.stdout
  rows = [("NAME", "SCALE", "TAGS", "SOURCE")]
  for name in registry.names():
    entry = registry.get(name)
    source = SOURCE_LABELS.get(type(entry.weights).__name__, "?")
    rows.append((name, f"x{entry.scale}", ", ".join(entry.tags) or "-", source))

  widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
  for row in rows:
    cells = [cell.ljust(widths[column]) for column, cell in enumerate(row)]
    print("  ".join(cells).rstrip(), file=stream)


def _run_upscale(args, parser):
  """Validate the upscale arguments and dispatch to the pipeline."""
  if args.scale is not None and args.size is not None:
    parser.error("use either --scale or --size, not both")

  try:
    upscale(args.input,
            args.output,
            scale=args.scale,
            target_size=args.size,
            model=args.model,
            device=args.device,
            fp16=args.fp16,
            tile=args.tile,
            tags=args.tags)
  except (RuntimeError, ValueError, KeyError) as error:
    parser.error(str(error))
  return 0


def main(argv=None):
  """Parse arguments and dispatch to the selected command."""
  argv = list(sys.argv[1:] if argv is None else argv)

  # A bare invocation should show the available commands rather than fall
  # through to the upscale command and complain about a missing input.
  if not argv:
    build_parser().print_help()
    return 0

  # Anything that is not an explicit subcommand is the default upscale command.
  # The help flags are left untouched so they reach the top level parser.
  if argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
    argv = ["upscale"] + argv

  parser = build_parser()
  args = parser.parse_args(argv)
  if args.command == "models":
    list_models()
    return 0
  return _run_upscale(args, parser)


if __name__ == "__main__":
  sys.exit(main())
