"""The model registry.

This is the single catalog of models srcli is allowed to run. Every entry is a
self contained recipe: how to build the network, how to obtain its weights, and
what native upscale factor those weights provide. Nothing about a model is
inferred at runtime, so there is no path by which the wrong architecture could
be paired with a set of tensors.

To add a model, add a builder in `builders` and an entry here. A future release
may also load entries from a user configuration file, but the built in catalog
below is always the base.
"""

from srcli.models import builders
from srcli.models.sources import HubFile, UrlFile

# Weights are pulled from the Hugging Face Hub where a maintained mirror exists,
# and from the original GitHub release assets otherwise. The GitHub release tags
# are the historical ones the models were published under, not the repository's
# latest release.
GITHUB = "https://github.com/xinntao/Real-ESRGAN/releases/download"


class ModelEntry:
  """A complete description of one runnable model.

  The three facts the pipeline needs are all here: `build` creates the network,
  `weights` locates and loads the tensors, and `scale` is the factor the model
  actually applies. `alignment` is the smallest multiple the network accepts for
  width and height, which the padding step must satisfy.
  """

  def __init__(self, name, scale, build, weights, alignment=None, tags=None):
    self.name = name
    self.scale = scale
    self.build = build
    self.weights = weights
    # Most networks only require the input to be divisible by their scale, which
    # is true for the RRDB family because of the pixel unshuffle at scale two.
    self.alignment = alignment if alignment is not None else scale
    self.tags = tags if tags is not None else []


REGISTRY = {
    "realesrgan-x4plus":
        ModelEntry(
            name="realesrgan-x4plus",
            scale=4,
            build=builders.build_rrdbnet_x4,
            weights=HubFile(repo_id="amd/realesrgan-x4plus",
                            filename="RealESRGAN_x4plus.pth"),
            tags=["photo", "general"],
        ),
    "realesrgan-x2plus":
        ModelEntry(
            name="realesrgan-x2plus",
            scale=2,
            build=builders.build_rrdbnet_x2,
            # The x2plus weights are not mirrored on the Hub under their own
            # repository, so this entry pulls from the original release asset.
            # Keeping it here also shows the URL source in use alongside the Hub.
            weights=UrlFile(url=f"{GITHUB}/v0.2.1/RealESRGAN_x2plus.pth",
                            filename="RealESRGAN_x2plus.pth"),
            tags=["photo", "general"],
        ),
    "realesrgan-x4plus-anime":
        ModelEntry(
            name="realesrgan-x4plus-anime",
            scale=4,
            build=builders.build_rrdbnet_anime,
            weights=HubFile(repo_id="amd/realesrgan-x4plus-anime-6b",
                            filename="RealESRGAN_x4plus_anime_6B.pth"),
            tags=["anime"],
        ),
}


def get(name):
  """Look up a model by name, raising a helpful error if it is unknown."""
  if name not in REGISTRY:
    raise KeyError("unknown model {!r}; available: {}".format(
        name, ", ".join(sorted(REGISTRY))))
  return REGISTRY[name]


def names():
  """Return the sorted list of registered model names."""
  return sorted(REGISTRY)
