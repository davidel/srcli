"""Tests for the command line layer."""

import io

import pytest

from srcli import cli


def test_list_models_prints_every_registered_model():
  buffer = io.StringIO()
  cli.list_models(stream=buffer)
  output = buffer.getvalue()
  for name in cli.registry.names():
    assert name in output
  # The header names the facts needed to choose a model.
  assert "NAME" in output
  assert "SCALE" in output
  assert "SOURCE" in output


def test_list_models_reports_scale_and_source():
  buffer = io.StringIO()
  cli.list_models(stream=buffer)
  lines = buffer.getvalue().splitlines()
  # Find the x4plus row and confirm the scale and hub source are shown.
  row = next(line for line in lines if line.startswith("realesrgan-x4plus "))
  assert "x4" in row
  assert "hf" in row


def test_list_models_labels_google_drive_weights():
  # The HAT checkpoints come from Google Drive, and the listing should make that
  # distinct from the Hub and direct URL sources.
  buffer = io.StringIO()
  cli.list_models(stream=buffer)
  lines = buffer.getvalue().splitlines()
  row = next(line for line in lines if line.startswith("hat-s-x4 "))
  assert "gdrive" in row
  assert "transformer" in row


def test_parser_routes_a_bare_invocation_to_upscale():
  parser = cli.build_parser()
  args = parser.parse_args(["upscale", "in.png", "-o", "out.png", "--scale", "4"])
  assert args.command == "upscale"
  assert args.scale == 4


def test_parser_accepts_the_models_subcommand():
  parser = cli.build_parser()
  assert parser.parse_args(["models"]).command == "models"


def test_parse_size_rejects_a_malformed_value():
  with pytest.raises(cli.argparse.ArgumentTypeError):
    cli.parse_size("3840-2160")
