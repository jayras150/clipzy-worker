"""End-to-end pipeline test on a real sample video.

FFmpeg-dependent stages run for real (download shim copies the synthetic
source; audio chunking, keyframes, scene detection, face scan, subtitle burn,
render, QC, manifest all execute). Only the external services are faked:
OpenRouter STT/Chat (canned payloads) and Cloudflare R2 (in-memory client).
"""

import json
import shutil

import pytest

from processors import downloader
from pipeline.runner import run_pipeline
from schemas.job_request import JobRequest
from services import resources
from services.progress import ProgressTracker
from engines import face_detection
from utils.logging_setup import JobLogger


@pytest.fixture()
def e2e_env(monkeypatch, sample_video, settings, fake_gpt, fake_stt, fake_r2, tmp_path):
    """Patch the three external boundaries: download, fonts/model, R2."""

    def fake_download(youtube_url, output_quality, cookies, workdir):
        workdir = type(workdir)(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample_video, workdir / "raw_video.mp4")
        from utils import ffmpeg as ff

        probe = ff.probe_media(workdir / "raw_video.mp4")
        width = next(int(s["width"]) for s in probe["streams"] if s["codec_type"] == "video")
        height = next(int(s["height"]) for s in probe["streams"] if s["codec_type"] == "video")
        duration = float(probe["format"]["duration"])
        return {"width": width, "height": height, "duration": duration,
                "size": (workdir / "raw_video.mp4").stat().st_size,
                "used_clients": "test_client", "format_selector": "test"}

    monkeypatch.setattr(downloader, "download_source", fake_download)
    monkeypatch.setattr(resources, "ensure_fonts", lambda *_: 0)
    monkeypatch.setattr(face_detection, "ensure_yunet_model", lambda *_: None)

    settings.r2_required = True
    settings.status_uploads = True
    settings.allow_nvenc = False
    return {"settings": settings, "r2": fake_r2, "gpt_calls": fake_gpt,
            "stt_calls": fake_stt}


def _request(**overrides):
    payload = {
        "job_id": "e2e_001",
        "youtube_url": "https://www.youtube.com/watch?v=testpattern",
        "target_duration": "5-25",
        "subtitle_style": "VIRAL",
        "subtitle_language": "",
        "aspect_ratio": "source",
        "preserve_source_quality": "true",
        "output_quality": "original",
        "manual_transform": None,
        "watermark_base64": "",
        "watermark_url": "",
        "watermark_position": "bottom_right",
        "watermark_size": "medium",
    }
    payload.update(overrides)
    return JobRequest.from_payload(payload)


def test_full_pipeline_completes(e2e_env, tmp_path):
    settings = e2e_env["settings"]
    r2 = e2e_env["r2"]

    progress = ProgressTracker("e2e_001")
    log = JobLogger("e2e_001", tmp_path / "job.log")
    response = run_pipeline(_request(), settings, progress, log, r2=r2)

    # ---- job response contract ------------------------------------------
    assert response["status"] == "completed"
    assert response["progress"] == 100
    assert response["clip_count"] == 2
    assert response["job_id"] == "e2e_001"
    assert response["download_url"].endswith("/outputs/e2e_001.mp4")
    assert response["manifest_url"].endswith("/outputs/e2e_001/manifest.json")
    assert response["error"] is None
    assert response["processing_time"] > 0
    assert response["video_language"] == "en"

    for clip in response["clips"]:
        assert clip["title"]
        assert 3 <= len(clip["tags"]) <= 5
        assert 1 <= clip["viral_score"] <= 100
        assert clip["render"]["video_codec"] == "h264"
        assert clip["render"]["width"] == 640
        assert clip["render"]["height"] == 360
        assert clip["subtitles"]["burned_in"] is True
        assert clip["subtitles"]["style"] == "VIRAL"
        assert clip["visual_analysis"]["faces_detected"] in (True, False)

    # Clip 1 carries the zoom intent -> editing metadata reflects it.
    clip1 = response["clips"][0]
    assert clip1["editing"]["zoom_enabled"] is True
    assert clip1["editing"]["zoom_cuts"]
    assert 1.05 <= clip1["editing"]["zoom_cuts"][0]["zoom_level"] <= 1.20

    # ---- R2 object layout (workflow parity) -------------------------------
    keys = set(r2.objects)
    assert "outputs/e2e_001/clip_1.mp4" in keys
    assert "outputs/e2e_001/clip_2.mp4" in keys
    assert "outputs/e2e_001.mp4" in keys
    assert "outputs/e2e_001_meta.json" in keys
    assert "outputs/e2e_001/manifest.json" in keys
    assert "status/e2e_001.json" in keys

    status_doc = json.loads(r2.objects["status/e2e_001.json"])
    assert status_doc["status"] == "ready"
    assert status_doc["job_id"] == "e2e_001"
    assert status_doc["clip_count"] == 2
    assert status_doc["subtitle_style"] == "VIRAL"
    assert status_doc["download_url"].endswith("/outputs/e2e_001.mp4")
    assert len(status_doc["clips"]) == 2

    # ---- the pipeline ran the real ffmpeg + AI (mocked endpoint) stages ----
    assert len(e2e_env["gpt_calls"]) >= 2  # curation + refinement
    assert len(e2e_env["stt_calls"]) >= 1  # speech audio chunk transcribed


def test_full_pipeline_manual_transform_and_aspect(e2e_env):
    settings = e2e_env["settings"]
    r2 = e2e_env["r2"]
    request = _request(job_id="e2e_002", aspect_ratio="9:16",
                       manual_transform={"scale": 1.4, "offset_x": 0.1, "offset_y": 0.0})
    progress = ProgressTracker("e2e_002")
    log = JobLogger("e2e_002", None)
    response = run_pipeline(request, settings, progress, log, r2=r2)

    assert response["status"] == "completed"
    for clip in response["clips"]:
        assert clip["render"]["width"] == 202 or clip["render"]["width"] == 204
        assert clip["render"]["height"] == 360
        mt = clip["editing"]["manual_transform"]
        assert mt["scale"] == 1.4 and mt["offset_x"] == 0.1


def test_full_pipeline_failure_uploads_debug(e2e_env, monkeypatch):
    settings = e2e_env["settings"]
    r2 = e2e_env["r2"]

    # Break the curation stage to force a failure path.
    from engines import viral_detection

    def broken_curate(*args, **kwargs):
        raise RuntimeError("simulated GPT-6 Luna outage")

    monkeypatch.setattr(viral_detection, "curate_clips", broken_curate)

    progress = ProgressTracker("e2e_003")
    log = JobLogger("e2e_003", None)
    with pytest.raises(Exception, match="simulated GPT-6 Luna outage"):
        run_pipeline(_request(job_id="e2e_003"), settings, progress, log, r2=r2)

    # Failure status.json uploaded with the error.
    assert "status/e2e_003.json" in r2.objects
    status_doc = json.loads(r2.objects["status/e2e_003.json"])
    assert status_doc["status"] == "failed"
    assert "simulated GPT-6 Luna outage" in status_doc["error"]
    # The isolated workspace is cleaned up (no temp video left behind).
    leftovers = [p for p in settings.work_root.rglob("*") if p.is_file()]
    assert leftovers == []


def test_job_cancellation_between_stages(e2e_env):
    settings = e2e_env["settings"]
    from pipeline.runner import JobCancelled

    progress = ProgressTracker("e2e_004")
    log = JobLogger("e2e_004", None)

    cancel_after = {"done": False}

    def cancel_check():
        # Cancel once the pipeline reaches the transcription stage.
        if progress._percent >= 34:
            cancel_after["done"] = True
        return cancel_after["done"]

    with pytest.raises(JobCancelled):
        run_pipeline(_request(job_id="e2e_004"), settings, progress, log,
                     cancel_check=cancel_check, r2=e2e_env["r2"])
    leftovers = [p for p in settings.work_root.rglob("*") if p.is_file()]
    assert leftovers == []
