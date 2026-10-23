import hashlib
import os
import sys
import tempfile

import pytest

from srcli.models import registry, sources
from srcli.models.sources import GDriveFile, HubFile, LocalFile, ensure_weights


def test_registry_entries_are_self_contained():
  # Every entry must name a builder, a weight source and a positive scale, since
  # those three facts are the whole contract the loader relies on.
  for name in registry.names():
    entry = registry.get(name)
    assert callable(entry.build)
    assert entry.weights is not None
    assert entry.scale > 0
    assert entry.alignment >= 1
    assert entry.name == name


def test_registry_get_reports_unknown_name_with_options():
  with pytest.raises(KeyError) as info:
    registry.get("does-not-exist")
  # The message should list what is available so a typo is easy to fix.
  assert "realesrgan-x4plus" in str(info.value)


def test_registry_builders_produce_the_declared_scale():
  # The declared scale and the scale baked into the network must agree, which is
  # what the loader and verify_scale ultimately check at runtime.
  for name in registry.names():
    entry = registry.get(name)
    module = entry.build()
    assert module.scale == entry.scale


def test_registry_offers_both_a_hub_and_a_url_source_style():
  # The catalog intentionally demonstrates both supported weight sources, so a
  # regression that dropped one style would show up here.
  kinds = {type(registry.get(name).weights).__name__ for name in registry.names()}
  assert "HubFile" in kinds
  assert "UrlFile" in kinds


def test_hat_entries_are_pinned_and_aligned_to_the_attention_window():
  # HAT attends over a sixteen pixel window, so every HAT entry must declare an
  # alignment of sixteen rather than defaulting to its scale. The weights come
  # from the official Google Drive folder rather than a repository, so each file
  # id and checksum must be pinned, since Drive offers no authoritative digest.
  hat = [name for name in registry.names() if name.startswith("hat-")]
  assert hat
  for name in hat:
    entry = registry.get(name)
    assert entry.alignment == 16
    assert isinstance(entry.weights, sources.GDriveFile)
    assert entry.weights.file_id
    assert entry.weights.sha256


def test_hat_small_and_large_share_a_scale_set():
  # Both widths should be available at every supported factor, so a caller can
  # trade quality for speed without losing the factor they asked for.
  scales = {"s": set(), "l": set()}
  for name in registry.names():
    if name.startswith("hat-s-"):
      scales["s"].add(registry.get(name).scale)
    elif name.startswith("hat-l-"):
      scales["l"].add(registry.get(name).scale)
  assert scales["s"] == {2, 3, 4}
  assert scales["s"] == scales["l"]


def test_hat_is_not_offered_for_automatic_selection():
  # HAT is heavy, so it deliberately omits the "general" tag and never enters
  # the automatic selection pool.
  for name in registry.names():
    if name.startswith("hat-"):
      assert "general" not in registry.get(name).tags


def test_local_file_reports_a_missing_path():
  source = LocalFile("/nonexistent/weights.pth")
  with pytest.raises(FileNotFoundError):
    source.fetch()


def test_local_file_returns_an_existing_path():
  handle, path = tempfile.mkstemp(suffix=".pth")
  os.close(handle)
  try:
    assert LocalFile(path).fetch() == path
  finally:
    os.remove(path)


def test_ensure_weights_reuses_a_cached_file(tmp_path, monkeypatch):
  # Point the cache at a temporary directory and pre-place the file, so no
  # network access happens and the cached copy is returned directly.
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
  os.makedirs(os.path.join(str(tmp_path), "srcli"), exist_ok=True)
  cached = os.path.join(str(tmp_path), "srcli", "cached.pth")
  with open(cached, "wb") as handle:
    handle.write(b"weights")

  result = ensure_weights("http://example.invalid/cached.pth", "cached.pth")
  assert result == cached


class _FakeResponse:
  """A minimal stand-in for the object urlopen returns.

  It behaves like a real stream: reads advance through the payload and return
  an empty block once exhausted, which is what copyfileobj relies on to stop.
  """

  def __init__(self, payload):
    self._payload = payload
    self._position = 0

  def __enter__(self):
    return self

  def __exit__(self, *args):
    return False

  def read(self, size=-1):
    if size is None or size < 0:
      size = len(self._payload) - self._position
    chunk = self._payload[self._position:self._position + size]
    self._position += len(chunk)
    return chunk


def test_ensure_weights_rejects_a_checksum_mismatch(tmp_path, monkeypatch):
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
  # Simulate a download that returns content whose hash does not match the pin,
  # so the check runs on freshly fetched bytes rather than on a cached file.
  monkeypatch.setattr(sources.urllib.request, "urlopen",
                      lambda url: _FakeResponse(b"tampered"))

  with pytest.raises(ValueError):
    ensure_weights("http://example.invalid/bad.pth",
                   "bad.pth",
                   expected_sha256="deadbeef")


def test_ensure_weights_verifies_and_stores_a_good_download(tmp_path, monkeypatch):
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
  payload = b"real weights"
  digest = hashlib.sha256(payload).hexdigest()
  monkeypatch.setattr(sources.urllib.request, "urlopen",
                      lambda url: _FakeResponse(payload))

  result = ensure_weights("http://example.invalid/good.pth",
                          "good.pth",
                          expected_sha256=digest)
  with open(result, "rb") as handle:
    assert handle.read() == payload


def _fake_hub_download(payload):
  """Return an hf_hub_download stand-in that writes `payload` to a temp file."""

  def _download(repo_id=None, filename=None, revision=None):
    handle, path = tempfile.mkstemp(suffix=".pth")
    with os.fdopen(handle, "wb") as out:
      out.write(payload)
    return path

  return _download


def test_hub_file_accepts_a_matching_checksum(monkeypatch):
  payload = b"hub weights"
  digest = hashlib.sha256(payload).hexdigest()
  monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hub_download(payload))

  source = HubFile(repo_id="org/repo", filename="model.pth", sha256=digest)
  path = source.fetch()
  with open(path, "rb") as handle:
    assert handle.read() == payload


def test_hub_file_rejects_a_checksum_mismatch(monkeypatch):
  monkeypatch.setattr("huggingface_hub.hf_hub_download",
                      _fake_hub_download(b"tampered"))

  source = HubFile(repo_id="org/repo", filename="model.pth", sha256="deadbeef")
  with pytest.raises(ValueError):
    source.fetch()


class _FakeGdown:
  """A stand-in for the gdown module that writes a fixed payload.

  gdown is an optional dependency, so the tests inject this into `sys.modules`
  rather than requiring it to be installed.
  """

  def __init__(self, payload, raises=None):
    self.payload = payload
    self.raises = raises

  def download(self, id=None, output=None, quiet=False):
    if self.raises is not None:
      raise self.raises
    with open(output, "wb") as handle:
      handle.write(self.payload)
    return output


def test_gdrive_file_accepts_a_matching_checksum(monkeypatch, tmp_path):
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
  payload = b"drive weights"
  digest = hashlib.sha256(payload).hexdigest()
  monkeypatch.setitem(sys.modules, "gdown", _FakeGdown(payload))

  source = GDriveFile(file_id="abc123", filename="model.pth", sha256=digest)
  path = source.fetch()
  with open(path, "rb") as handle:
    assert handle.read() == payload


def test_gdrive_file_rejects_a_checksum_mismatch(monkeypatch, tmp_path):
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
  monkeypatch.setitem(sys.modules, "gdown", _FakeGdown(b"tampered"))

  source = GDriveFile(file_id="abc123", filename="model.pth", sha256="deadbeef")
  with pytest.raises(ValueError):
    source.fetch()


def test_gdrive_file_reports_a_missing_dependency(monkeypatch, tmp_path):
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
  # Setting the module to None makes `import gdown` raise ImportError.
  monkeypatch.setitem(sys.modules, "gdown", None)

  source = GDriveFile(file_id="abc123", filename="model.pth")
  with pytest.raises(ImportError):
    source.fetch()


def test_gdrive_file_leaves_no_partial_file_on_failure(monkeypatch, tmp_path):
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
  monkeypatch.setitem(sys.modules, "gdown",
                      _FakeGdown(b"x", raises=RuntimeError("quota exceeded")))

  source = GDriveFile(file_id="abc123", filename="model.pth")
  with pytest.raises(RuntimeError):
    source.fetch()
  # Nothing should be committed to the cache after a failed download.
  assert not os.path.exists(os.path.join(str(tmp_path), "srcli", "model.pth"))
