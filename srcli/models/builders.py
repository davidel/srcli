"""Architecture constructors for the registry.

Each builder returns a fresh, untrained network. The registry pairs a builder
with a set of weights, so the architecture is always something we construct
ourselves rather than something guessed from a checkpoint. That is what makes
loading deterministic: there is no detection step that could pick the wrong
network for a given set of tensors.
"""

from srcli.models.hat import HAT
from srcli.models.rrdbnet import RRDBNet

# The HAT body comes in two widths. Both use six attention heads per group and a
# sixteen pixel attention window; the large variant simply stacks twice as many
# groups. These are the values the released ImageNet-pretrained checkpoints were
# trained with, and they are not interchangeable between widths.
_HAT_SMALL = {
    "embed_dim": 180,
    "depths": (6, 6, 6, 6, 6, 6),
    "num_heads": (6, 6, 6, 6, 6, 6),
}

_HAT_LARGE = {
    "embed_dim": 180,
    "depths": (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    "num_heads": (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
}


def _build_hat(config, scale):
  """Build a HAT generator for one scale from a width configuration."""
  return HAT(in_chans=3,
             window_size=16,
             compress_ratio=3,
             squeeze_factor=30,
             conv_scale=0.01,
             overlap_ratio=0.5,
             mlp_ratio=2,
             upsampler="pixelshuffle",
             resi_connection="1conv",
             upscale=scale,
             **config)


def build_hat_s_x2():
  """HAT-S x2, the small ImageNet-pretrained generator."""
  return _build_hat(_HAT_SMALL, 2)


def build_hat_s_x3():
  """HAT-S x3, the small ImageNet-pretrained generator."""
  return _build_hat(_HAT_SMALL, 3)


def build_hat_s_x4():
  """HAT-S x4, the small ImageNet-pretrained generator."""
  return _build_hat(_HAT_SMALL, 4)


def build_hat_l_x2():
  """HAT-L x2, the large ImageNet-pretrained generator."""
  return _build_hat(_HAT_LARGE, 2)


def build_hat_l_x3():
  """HAT-L x3, the large ImageNet-pretrained generator."""
  return _build_hat(_HAT_LARGE, 3)


def build_hat_l_x4():
  """HAT-L x4, the large ImageNet-pretrained generator."""
  return _build_hat(_HAT_LARGE, 4)


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
