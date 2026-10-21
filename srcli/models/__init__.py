"""Model registry, weight sources and loading.

This package is the only place that knows how to turn a model name into a
runnable network. It deliberately avoids any architecture detection: every model
is described by a registry entry that names its builder, its weight source and
its native upscale factor.
"""

from srcli.models.loader import LoadedModel, load_by_name, load_entry, verify_scale
from srcli.models.registry import ModelEntry, get, names

__all__ = [
    "LoadedModel",
    "ModelEntry",
    "get",
    "load_by_name",
    "load_entry",
    "names",
    "verify_scale",
]
