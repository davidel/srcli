"""srcli: a command line super-resolution tool.

The package is deliberately importable as a library. The CLI is only a thin
wrapper around the pipeline, so the same upscaling logic can be reused from
scripts without going through argument parsing.
"""

__version__ = "0.1.0"
