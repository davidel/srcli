"""The RRDBNet generator used by Real-ESRGAN.

This is a faithful re-implementation of the architecture from BasicSR, kept
local so the package does not drag in the whole BasicSR dependency tree just to
build one network. The module and parameter names must match the reference
exactly, because the published weights are loaded with strict state dict
loading and any rename would break that.
"""

import torch
from torch import nn
from torch.nn import functional as F


def make_layer(block, count, **kwargs):
  """Stack `count` copies of a block, threading the constructor kwargs."""
  layers = []
  for _ in range(count):
    layers.append(block(**kwargs))
  return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
  """Densely connected residual block with five convolutions."""

  def __init__(self, num_feat=64, num_grow_ch=32):
    super().__init__()
    self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
    self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
    self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
    self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
    self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x1 = self.lrelu(self.conv1(x))
    x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
    x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
    x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
    x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
    # The 0.2 residual scaling is part of the trained design, not a tuning knob.
    return x5 * 0.2 + x


class RRDB(nn.Module):
  """Residual in Residual Dense Block: three dense blocks with a skip."""

  def __init__(self, num_feat, num_grow_ch=32):
    super().__init__()
    self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
    self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
    self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

  def forward(self, x):
    out = self.rdb1(x)
    out = self.rdb2(out)
    out = self.rdb3(out)
    return out * 0.2 + x


class RRDBNet(nn.Module):
  """The generator network.

  The `scale` argument changes more than the final upsampling. At scale 2 the
  network folds the input down with a pixel unshuffle first, which is why the
  input width and height must be even. That requirement is surfaced through the
  backend's `alignment` so the padding step can satisfy it.
  """

  def __init__(self,
               num_in_ch,
               num_out_ch,
               scale=4,
               num_feat=64,
               num_block=23,
               num_grow_ch=32):
    super().__init__()
    self.scale = scale
    if scale == 2:
      num_in_ch = num_in_ch * 4
    elif scale == 1:
      num_in_ch = num_in_ch * 16
    self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
    self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
    self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    if self.scale == 2:
      feat = F.pixel_unshuffle(x, 2)
    elif self.scale == 1:
      feat = F.pixel_unshuffle(x, 4)
    else:
      feat = x
    feat = self.conv_first(feat)
    body_feat = self.conv_body(self.body(feat))
    feat = feat + body_feat
    feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2,
                                                  mode="nearest")))
    feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2,
                                                  mode="nearest")))
    out = self.conv_last(self.lrelu(self.conv_hr(feat)))
    return out
