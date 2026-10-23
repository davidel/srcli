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
from srcli.models.sources import GDriveFile, HubFile, UrlFile

# Weights are pulled from the Hugging Face Hub where a maintained mirror exists,
# from the original GitHub release assets otherwise, and from the official
# Google Drive folder for models that have no other maintained home. The GitHub
# release tags are the historical ones the models were published under, not the
# repository's latest release.
GITHUB = "https://github.com/xinntao/Real-ESRGAN/releases/download"

# HAT has no official Hugging Face repository. The upstream project publishes
# its checkpoints through a public Google Drive folder, and the file ids below
# are the ones from that folder. Drive gives no authoritative digest, so each
# sha256 was computed from a download of the official folder and pinned by hand;
# it catches a later swap or corruption on Drive, not a first-download forgery.
HAT_ALIGNMENT = 16


def _hat_entry(name, scale, build, filename, file_id, digest, rank):
  """Build a registry entry for an official HAT checkpoint on Google Drive."""
  return ModelEntry(
      name=name,
      scale=scale,
      build=build,
      weights=GDriveFile(file_id=file_id, filename=filename, sha256=digest),
      alignment=HAT_ALIGNMENT,
      # HAT is a large transformer and markedly heavier than the RRDB family, so
      # it deliberately omits the "general" tag that `select` uses to build its
      # default pool. It is chosen by name or by an explicit tag, never by
      # accident, which keeps the automatic path fast and predictable.
      tags=["photo", "transformer"],
      # The small width is rank zero so it wins a tie against the large width,
      # which is rank one. Both widths are reachable by name either way.
      rank=rank,
  )


class ModelEntry:
  """A complete description of one runnable model.

  The three facts the pipeline needs are all here: `build` creates the network,
  `weights` locates and loads the tensors, and `scale` is the factor the model
  actually applies. `alignment` is the smallest multiple the network accepts for
  width and height, which the padding step must satisfy.
  """

  def __init__(self, name, scale, build, weights, alignment=None, tags=None, rank=0):
    self.name = name
    self.scale = scale
    self.build = build
    self.weights = weights
    # Most networks only require the input to be divisible by their scale, which
    # is true for the RRDB family because of the pixel unshuffle at scale two.
    self.alignment = alignment if alignment is not None else scale
    self.tags = tags if tags is not None else []
    # When several entries offer the same factor, selection prefers the lowest
    # rank and only then falls back to the name. It is how a heavier variant of
    # the same family is kept from winning a tie by accident of alphabetical
    # order, so the cheaper model is the default and the larger one is opt-in.
    self.rank = rank


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
    "hat-s-x2":
        _hat_entry(
            name="hat-s-x2",
            scale=2,
            build=builders.build_hat_s_x2,
            filename="HAT_SRx2_ImageNet-pretrain.pth",
            file_id="11WDyK4MMcRapHs_aKJKHaYAFsb29SoCw",
            digest="82ebd911263bcc886fbef46b30cf97b92a932a27a3cba30163d4577afb09b9d7",
            rank=0,
        ),
    "hat-s-x3":
        _hat_entry(
            name="hat-s-x3",
            scale=3,
            build=builders.build_hat_s_x3,
            filename="HAT_SRx3_ImageNet-pretrain.pth",
            file_id="1ugC8yyya4Gcvlo01rfGYRFjcpXJ-U-Sm",
            digest="8469db903d464497f94419587054c9e83dff994edc409d563ab8f5a503767be8",
            rank=0,
        ),
    "hat-s-x4":
        _hat_entry(
            name="hat-s-x4",
            scale=4,
            build=builders.build_hat_s_x4,
            filename="HAT_SRx4_ImageNet-pretrain.pth",
            file_id="1cxls85ZE7kalhNy47eBJI_L_Lwf9hxRI",
            digest="4ee053c42461187846dc0e93aa5abd34591c0725a8e044a59000e92ee215e833",
            rank=0,
        ),
    "hat-l-x2":
        _hat_entry(
            name="hat-l-x2",
            scale=2,
            build=builders.build_hat_l_x2,
            filename="HAT-L_SRx2_ImageNet-pretrain.pth",
            file_id="16xtMezHvckdWEuSiOxcO-dgOlsI0rEUg",
            digest="2818c7ca8d72ec4cc5f31c93203d55252a662dd35cda34ce1a69661f97dcd38f",
            rank=1,
        ),
    "hat-l-x3":
        _hat_entry(
            name="hat-l-x3",
            scale=3,
            build=builders.build_hat_l_x3,
            filename="HAT-L_SRx3_ImageNet-pretrain.pth",
            file_id="1f0iVM1coyVx5KffUYWw9g4JB1IG0QCEc",
            digest="78af181bedf1e805fd7517d3738bc5824f7ebdc477fc9f757b708bcf49ad4e3d",
            rank=1,
        ),
    "hat-l-x4":
        _hat_entry(
            name="hat-l-x4",
            scale=4,
            build=builders.build_hat_l_x4,
            filename="HAT-L_SRx4_ImageNet-pretrain.pth",
            file_id="1uefIctjoNE3Tg6GTzelesTTshVogQdUf",
            digest="5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043",
            rank=1,
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
