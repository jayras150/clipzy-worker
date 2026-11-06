"""Cache manager tests: validation, TTL, atomic writes, path scoping."""

import json
import time

from cache import cache_manager


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "c" / "entry.json"
    cache_manager.save_cached(path, {"result": 42}, meta={"source_sha256": "abc", "model": "m"})
    loaded = cache_manager.load_cached(path, expect={"source_sha256": "abc", "model": "m"})
    assert loaded["result"] == 42
    assert "cache_meta" in loaded
    assert loaded["cache_meta"]["created_at"] > 0


def test_meta_mismatch_rejects(tmp_path):
    path = tmp_path / "entry.json"
    cache_manager.save_cached(path, {"result": 1}, meta={"source_sha256": "video1"})
    assert cache_manager.load_cached(path, expect={"source_sha256": "video2"}) is None
    # A DIFFERENT video can never reuse another video's computation.
    assert cache_manager.load_cached(path, expect={"source_sha256": "video1"}) is not None


def test_ttl_expiry(tmp_path):
    path = tmp_path / "entry.json"
    cache_manager.save_cached(path, {"result": 1}, meta={"k": "v"})
    # Fake an old entry by rewriting created_at.
    data = json.loads(path.read_text())
    data["cache_meta"]["created_at"] = time.time() - 40 * 86400
    path.write_text(json.dumps(data))
    assert cache_manager.load_cached(path, expect={"k": "v"}, ttl_days=30) is None
    assert cache_manager.load_cached(path, expect={"k": "v"}) is not None  # no TTL check


def test_corrupt_or_missing_entries(tmp_path):
    assert cache_manager.load_cached(tmp_path / "nope.json", expect={}) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{broken")
    assert cache_manager.load_cached(bad, expect={}) is None
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2]")
    assert cache_manager.load_cached(arr, expect={}) is None


def test_cache_paths_are_hash_scoped(tmp_path):
    paths = cache_manager.CachePaths(tmp_path, "sha123")
    assert paths.transcript().endswith("transcript/sha123/transcript.json")
    assert paths.curation().endswith("curation/sha123/curation.json")
    assert paths.face_context().endswith("vision/sha123/face_context.json")
    other = cache_manager.CachePaths(tmp_path, "sha999")
    assert other.transcript() != paths.transcript()
