"""Architecture constructors for the registry.

Each builder returns a fresh, untrained network. The registry pairs a builder
with a set of weights, so the architecture is always something we construct
ourselves rather than something guessed from a checkpoint. That is what makes
loading deterministic: there is no detection step that could pick the wrong
network for a given set of tensors.
"""

from srcli.models.rrdbnet import RRDBNet


def build_rrdbnet_x4():
  """Real-ESRGAN x4plus: the standard 23 block RRDBNet generator."""
  return RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_block=23)


def build_rrdbnet_x2():
  """Real-ESRGAN x2plus.

  Same body as x4 but with a native factor of two, which makes the network fold
  the input down with a pixel unshuffle first and therefore requires even input
  dimensions.
  """
  return RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_block=23)
