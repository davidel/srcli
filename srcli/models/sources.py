"""Weight sources.

A source answers one question: where do the weight tensors live, and how do we
get them onto local disk. Fetching is kept separate from architecture and from
loading so a new location (a mirror, an internal store) only needs a new class
here, with no change to the builders or the loader.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request

# The common sub-dictionary keys under which released checkpoints wrap their
# tensors. Real-ESRGAN ships both a raw and an EMA set; the EMA one is the set
# that was evaluated, so it is the default choice.
DEFAULT_STATE_KEY = "params_ema"


def cache_dir():
  """Return the directory where downloaded weights are stored."""
  base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"),
                                                          ".cache")
  return os.path.join(base, "srcli")


def _sha256(path):
  """Hash a file in chunks so a large model does not sit in memory."""
  digest = hashlib.sha256()
  with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def ensure_weights(url, filename, expected_sha256=None):
  """Return a local path to the weights, downloading them if necessary.

  If `expected_sha256` is given the file is verified after download and before
  use, so a truncated or tampered download fails loudly instead of producing
  garbage. The download lands in a temporary file first and is only moved into
  place once complete, so an interrupted run never leaves a half written file
  that a later run would mistake for a finished download.
  """
  target = os.path.join(cache_dir(), filename)
  if os.path.exists(target) and (expected_sha256 is None or
                                 _sha256(target) == expected_sha256):
    return target

  os.makedirs(cache_dir(), exist_ok=True)
  handle, temp_path = tempfile.mkstemp(dir=cache_dir(), suffix=".part")
  os.close(handle)
  try:
    with urllib.request.urlopen(url) as response, open(temp_path, "wb") as out:
      shutil.copyfileobj(response, out)
    if expected_sha256 is not None:
      actual = _sha256(temp_path)
      if actual != expected_sha256:
        raise ValueError(
            f"checksum mismatch for {filename}: expected {expected_sha256}, got {actual}"
        )
    os.replace(temp_path, target)
  except BaseException:
    # Clean up the partial file on any failure, including an interrupt.
    if os.path.exists(temp_path):
      os.remove(temp_path)
    raise
  return target


class LocalFile:
  """Weights that are already on disk. Nothing is downloaded."""

  def __init__(self, path, state_key=DEFAULT_STATE_KEY):
    self.path = path
    self.state_key = state_key

  def fetch(self):
    if not os.path.exists(self.path):
      raise FileNotFoundError(f"weights not found: {self.path}")
    return self.path


class UrlFile:
  """Weights at a direct HTTP(S) link, cached on disk by filename."""

  def __init__(self, url, filename, state_key=DEFAULT_STATE_KEY, sha256=None):
    self.url = url
    self.filename = filename
    self.state_key = state_key
    self.sha256 = sha256

  def fetch(self):
    return ensure_weights(self.url, self.filename, self.sha256)


class HubFile:
  """Weights on the Hugging Face Hub.

  The Hub client brings its own content addressed cache, authentication for
  gated repositories and resumable downloads, so none of that is reimplemented
  here. The import is deferred so that a user who never touches the Hub is not
  required to install the client.
  """

  def __init__(self, repo_id, filename, revision="main", state_key=DEFAULT_STATE_KEY):
    self.repo_id = repo_id
    self.filename = filename
    self.revision = revision
    self.state_key = state_key

  def fetch(self):
    try:
      from huggingface_hub import hf_hub_download
    except ImportError as error:
      raise ImportError(
          "this model is hosted on the Hugging Face Hub; install the optional "
          "dependency with: pip install 'srcli[hf]'") from error
    return hf_hub_download(repo_id=self.repo_id,
                           filename=self.filename,
                           revision=self.revision)
