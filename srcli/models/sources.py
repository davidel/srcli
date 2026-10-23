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


def _download_cached(filename, download, expected_sha256=None):
  """Cache a download by filename, verifying it before it is committed.

  `download` is a callable that writes the file to a path it is given. The
  download lands in a temporary file first and is only moved into place once it
  is complete and, when a digest is pinned, verified. A truncated or tampered
  download therefore fails loudly instead of producing garbage, and an
  interrupted run never leaves a half written file that a later run would
  mistake for a finished download.
  """
  target = os.path.join(cache_dir(), filename)
  if os.path.exists(target) and (expected_sha256 is None or
                                 _sha256(target) == expected_sha256):
    return target

  os.makedirs(cache_dir(), exist_ok=True)
  handle, temp_path = tempfile.mkstemp(dir=cache_dir(), suffix=".part")
  os.close(handle)
  try:
    download(temp_path)
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


def ensure_weights(url, filename, expected_sha256=None):
  """Return a local path to the weights at `url`, downloading if necessary."""

  def download(temp_path):
    with urllib.request.urlopen(url) as response, open(temp_path, "wb") as out:
      shutil.copyfileobj(response, out)

  return _download_cached(filename, download, expected_sha256)


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


class GDriveFile:
  """Weights shared through a public Google Drive file.

  Drive does not serve the bytes directly: a share link answers with an HTML
  interstitial and a redirect that a plain HTTP client cannot follow reliably.
  `gdown` handles that dance, so it is the only way in. The import is deferred so
  a user who never touches Drive is not required to install it.

  Google only publishes an MD5 for shared files, never a SHA256, so the digest
  here is one we computed from the official folder and pinned by hand. It cannot
  prove the first download was genuine, but it does mean a later swap or
  corruption on Drive is caught rather than run.
  """

  def __init__(self, file_id, filename, state_key=DEFAULT_STATE_KEY, sha256=None):
    self.file_id = file_id
    self.filename = filename
    self.state_key = state_key
    self.sha256 = sha256

  def fetch(self):
    try:
      import gdown
    except ImportError as error:
      raise ImportError("this model is hosted on Google Drive; install the optional "
                        "dependency with: pip install 'srcli[gdrive]'") from error

    def download(temp_path):
      # gdown raises on failure in recent releases, so a bad id or an exceeded
      # quota surfaces as an error rather than a silently empty file.
      gdown.download(id=self.file_id, output=temp_path, quiet=True)

    return _download_cached(self.filename, download, self.sha256)


class HubFile:
  """Weights on the Hugging Face Hub.

  The Hub client brings its own content addressed cache, authentication for
  gated repositories and resumable downloads, so none of that is reimplemented
  here. The import is deferred so that a user who never touches the Hub is not
  required to install the client.
  """

  def __init__(self,
               repo_id,
               filename,
               revision="main",
               state_key=DEFAULT_STATE_KEY,
               sha256=None):
    self.repo_id = repo_id
    self.filename = filename
    self.revision = revision
    self.state_key = state_key
    self.sha256 = sha256

  def fetch(self):
    try:
      from huggingface_hub import hf_hub_download
    except ImportError as error:
      raise ImportError(
          "this model is hosted on the Hugging Face Hub; install the optional "
          "dependency with: pip install 'srcli[hf]'") from error
    path = hf_hub_download(repo_id=self.repo_id,
                           filename=self.filename,
                           revision=self.revision)
    if self.sha256 is not None:
      # The Hub client already checks the file against its own manifest, but a
      # pinned digest is what makes the mirror choice auditable: if the upstream
      # repository is retagged or replaced we want to fail rather than silently
      # run a different network.
      actual = _sha256(path)
      if actual != self.sha256:
        raise ValueError(
            f"checksum mismatch for {self.filename}: expected {self.sha256}, "
            f"got {actual}")
    return path
