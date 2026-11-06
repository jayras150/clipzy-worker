"""Downloader internals, encoder selection, workspace isolation, worker thread."""

import json

import pytest

from processors import downloader
from services.progress import ProgressTracker
from storage.file_manager import JobWorkspace
from utils import ffmpeg as ff
from utils import gpu


def test_format_selector_policy():
    assert downloader._format_selector("original") == "bv*[height<=2160]+ba/bv*+ba/b"
    assert downloader._format_selector("1080p") == "bv*[height<=1080]+ba/bv*+ba/b"
    assert downloader._format_selector("") == "bv*[height<=2160]+ba/bv*+ba/b"


def test_client_sets_cookie_aware():
    primary, fallback = downloader._client_sets(True)
    # Cookie-authenticated runs must only use cookie-compatible clients.
    assert "ios" not in primary and "android" not in primary and "visionos" not in primary
    assert primary == "web_embedded,tv_samsung,web"
    assert fallback == "tv,tv_samsung,web_embedded,web,web_safari,mweb"

    primary_anon, fallback_anon = downloader._client_sets(False)
    assert primary_anon == "tv_samsung,visionos,web"
    assert "ios" in fallback_anon and "android" in fallback_anon


def test_cookie_file_written_with_0600(tmp_path):
    flags = downloader._write_cookies(tmp_path / "cookies.txt", "NETSCAPE 1.0\n...")
    assert flags == ["--cookies", str(tmp_path / "cookies.txt")]
    import os

    # Windows filesystems do not expose POSIX 0600 mode bits through st_mode.
    # The production chmod remains in place; verify the exact mode on POSIX.
    if os.name != "nt":
        assert os.stat(tmp_path / "cookies.txt").st_mode & 0o777 == 0o600


def test_download_fails_loudly_without_video(tmp_path, monkeypatch):
    # yt-dlp "succeeds" but produces nothing: verification must fail the job.
    import subprocess

    def fake_run(cmd, capture_output=True, text=True, cwd=None, timeout=None):
        return subprocess.CompletedProcess(cmd, 0, stdout="no video", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(downloader.DownloadError, match="yt-dlp failed to produce"):
        downloader.download_source("https://example.com/x", "original", "", tmp_path)


def test_choose_encoder_cpu_fallback(monkeypatch):
    monkeypatch.setattr(ff, "has_nvenc", lambda: False)
    encoder = ff.choose_encoder(allow_nvenc=True)
    assert encoder["codec"] == "libx264"
    assert encoder["quality_flags"] == ["-crf", "18"]

    encoder = ff.choose_encoder(allow_nvenc=False)
    assert encoder["codec"] == "libx264"


def test_choose_encoder_nvenc(monkeypatch):
    monkeypatch.setattr(ff, "has_nvenc", lambda: True)
    encoder = ff.choose_encoder(allow_nvenc=True)
    assert encoder["codec"] == "h264_nvenc"
    assert encoder["gpu"] is True
    assert "-cq" in encoder["quality_flags"]


def test_gpu_detection_without_gpu():
    info = gpu.detect_gpu()
    assert info.available is False  # sandbox: no NVIDIA GPU
    assert gpu.gpu_memory_used_mb() is None


def test_workspace_isolation_and_cleanup(tmp_path):
    ws1 = JobWorkspace(tmp_path, "jobA").create()
    ws2 = JobWorkspace(tmp_path, "jobB").create()
    assert ws1.root != ws2.root
    ws1.source.write_bytes(b"video A")
    ws2.source.write_bytes(b"video B")
    # Unrelated jobs never share temporary video files.
    assert ws1.source.read_bytes() != ws2.source.read_bytes()

    ws1.cleanup()
    assert not ws1.root.exists()
    assert ws2.root.exists()  # sibling untouched
    ws2.cleanup()


def test_progress_stage_sequence():
    seen = []
    tracker = ProgressTracker("j", on_update=lambda s, l, p: seen.append((s, p)))
    for stage in ("validate", "download", "transcribe", "upload"):
        tracker.stage(stage)
    assert seen[0] == ("validate", 2)
    assert seen[1] == ("download", 12)
    assert seen[-1] == ("upload", 100)
    snapshot = tracker.snapshot()
    assert snapshot["progress"] == 100
    assert snapshot["stage_label"]


def test_job_response_and_status_documents():
    from schemas.job_response import build_job_response, build_status_document

    manifest = [{"clip_index": 1}, {"clip_index": 2}]
    params = {"subtitle_style": "VIRAL", "aspect_ratio": "9:16",
              "output_quality": "original", "manual_transform": None}
    status = build_status_document("job1", "ready", "https://cdn.x", manifest, params)
    assert status["job_id"] == "job1"
    assert status["status"] == "ready"
    assert status["file_name"] == "job1.mp4"
    assert status["download_url"] == "https://cdn.x/outputs/job1.mp4"
    assert status["manifest_url"] == "https://cdn.x/outputs/job1/manifest.json"
    assert status["clip_count"] == 2
    assert len(status["clips"]) == 2

    response = build_job_response("job1", "completed", manifest, manifest,
                                   "https://cdn.x", 120.5, params=params)
    assert response["status"] == "completed"
    assert response["processing_time"] == 120.5
    assert response["error"] is None


def test_worker_pipeline_server_thread(tmp_path, monkeypatch):
    """The PyWorker's internal model server starts, reports readiness in the
    model log file (the LogActionConfig on_load contract), and serves jobs."""
    import worker as worker_mod
    from config import WorkerSettings
    from services import resources
    from engines import face_detection

    monkeypatch.setenv("CLIPZY_WORK_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("CLIPZY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLIPZY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CLIPZY_FONTS_DIR", str(tmp_path / "fonts"))
    monkeypatch.setenv("CLIPZY_PIPELINE_LOG_FILE", str(tmp_path / "pipeline.log"))
    monkeypatch.setenv("CLIPZY_PIPELINE_PORT", "18099")
    settings = WorkerSettings()
    settings.ensure_dirs()

    monkeypatch.setattr(resources, "ensure_fonts", lambda *_: 0)
    monkeypatch.setattr(face_detection, "ensure_yunet_model", lambda *_: None)

    worker_mod._start_pipeline_server(settings)
    assert worker_mod._wait_for_pipeline(settings, timeout=30) is True

    import urllib.request

    with urllib.request.urlopen("http://127.0.0.1:18099/health", timeout=5) as response:
        assert response.status == 200
    probe_body = json.dumps({"probe": True}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:18099/probe", data=probe_body, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        doc = json.loads(response.read())
        assert doc["ok"] is True and "gpu" in doc

    log_text = (tmp_path / "pipeline.log").read_text()
    assert "Application startup complete." in log_text  # on_load contract
    assert "Clipzy pipeline server starting." in log_text
