"""R2 client: object layout, verification, retries (boto3 mocked)."""

import json

import pytest

from storage import r2_client


class FakeBoto3Client:
    def __init__(self, fail_times=0, size_mismatch=False):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.put_calls = 0
        self.fail_times = fail_times
        self.size_mismatch = size_mismatch

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.put_calls += 1
        if self.put_calls <= self.fail_times:
            raise ConnectionError("simulated R2 hiccup")
        self.objects[Key] = (Body.read() if hasattr(Body, "read") else Body, ContentType)

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        body = self.objects[Key][0]
        size = 1 if self.size_mismatch else len(body)
        return {"ContentLength": size}


@pytest.fixture()
def r2(tmp_path, monkeypatch):
    fake = FakeBoto3Client()

    def fake_boto_client(*args, **kwargs):
        return fake

    import boto3
    monkeypatch.setattr(boto3, "client", fake_boto_client)
    client = r2_client.R2Client("https://acct.r2.cloudflarestorage.com", "key", "secret",
                                "bucket", "https://cdn.example.com", "auto")
    client._fake = fake
    return client


def test_upload_file_verifies_and_returns_url(r2, tmp_path):
    path = tmp_path / "clip_1.mp4"
    path.write_bytes(b"\x00" * 2048)
    url = r2.upload_file(path, "outputs/job/clip_1.mp4", "video/mp4")
    assert url == "https://cdn.example.com/outputs/job/clip_1.mp4"
    assert "outputs/job/clip_1.mp4" in r2._fake.objects


def test_upload_file_retries_on_transient_failures(r2, tmp_path):
    r2._fake.fail_times = 2
    path = tmp_path / "clip_2.mp4"
    path.write_bytes(b"\x00" * 2048)
    from unittest.mock import patch
    with patch.object(r2_client.time, "sleep"):
        url = r2.upload_file(path, "outputs/job/clip_2.mp4")
    assert url
    assert r2._fake.put_calls == 3  # 2 failures + 1 success


def test_upload_file_raises_after_exhausted_retries(r2, tmp_path):
    r2._fake.fail_times = 99
    path = tmp_path / "clip_3.mp4"
    path.write_bytes(b"\x00" * 2048)
    from unittest.mock import patch
    with patch.object(r2_client.time, "sleep"):
        with pytest.raises(r2_client.R2Error, match="after 4 attempts"):
            r2.upload_file(path, "outputs/job/clip_3.mp4")


def test_upload_verification_detects_size_mismatch(r2, tmp_path):
    r2._fake.size_mismatch = True
    path = tmp_path / "clip_4.mp4"
    path.write_bytes(b"\x00" * 4096)
    with pytest.raises(r2_client.R2Error, match="verification failed"):
        r2.upload_file(path, "outputs/job/clip_4.mp4")


def test_upload_job_outputs_layout(r2, tmp_path):
    clips_dir = tmp_path / "output_clips"
    clips_dir.mkdir()
    (clips_dir / "clip_1.mp4").write_bytes(b"\x00" * 2048)
    (clips_dir / "clip_2.mp4").write_bytes(b"\x00" * 2048)
    manifest = [{"clip_index": 1}, {"clip_index": 2}]

    urls = r2.upload_job_outputs("jobX", clips_dir, manifest)

    keys = set(r2._fake.objects)
    assert "outputs/jobX/clip_1.mp4" in keys
    assert "outputs/jobX/clip_2.mp4" in keys
    assert "outputs/jobX.mp4" in keys                      # primary (first clip)
    assert "outputs/jobX_meta.json" in keys                # Android legacy location
    assert "outputs/jobX/manifest.json" in keys            # folder manifest
    assert urls["primary"] == "https://cdn.example.com/outputs/jobX.mp4"
    assert urls["manifest"].endswith("outputs/jobX/manifest.json")
    # Manifest content preserved verbatim.
    body = r2._fake.objects["outputs/jobX_meta.json"][0]
    assert json.loads(body.decode() if isinstance(body, bytes) else body) == manifest


def test_upload_job_outputs_requires_clips(r2, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(r2_client.R2Error, match="no rendered clip"):
        r2.upload_job_outputs("jobY", empty, [])


def test_upload_status_and_debug(r2, tmp_path):
    url = r2.upload_status("jobZ", {"job_id": "jobZ", "status": "ready"})
    assert url.endswith("status/jobZ.json")
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    (debug_dir / "curation_attempt_1.txt").write_text("attempt data")
    urls = r2.upload_debug("jobZ", debug_dir, [])
    assert any(u.endswith("outputs/jobZ/debug/curation_attempt_1.txt") for u in urls)
