"""Job response schema.

Two output shapes exist, both preserved from the workflow:

1. `status.json` — the R2 object at status/<job_id>.json the Android app polls.
   Byte-compatible with the workflow's PY_STATUS output, extended with a
   `progress` field and intermediate `processing` states (final "ready"
   documents keep every original key).
2. The full job response — what the worker returns through the Vast.ai
   request envelope when the job completes (docs/API.md, section 11).
"""

from __future__ import annotations

import time
from typing import Any


def build_status_document(job_id: str, status: str, public_domain: str, manifest: list,
                          params: dict, transcript_language: str = "",
                          progress: int = 0, error: str | None = None,
                          stage: str = "") -> dict:
    """status.json document (same key set as the workflow, plus progress)."""
    domain = (public_domain or "").rstrip("/")
    clips = [
        {
            "clip_index": clip["clip_index"],
            "file": f"clip_{clip['clip_index']}.mp4",
            "url": f"{domain}/outputs/{job_id}/clip_{clip['clip_index']}.mp4",
        }
        for clip in manifest
    ] if manifest else []

    document: dict[str, Any] = {
        "job_id": job_id,
        "status": status,  # processing | ready | failed
        "progress": progress,
        "file_name": f"{job_id}.mp4",
        "download_url": f"{domain}/outputs/{job_id}.mp4",
        "manifest_url": f"{domain}/outputs/{job_id}/manifest.json",
        "clip_count": len(manifest),
        "subtitle_style": params.get("subtitle_style", "VIRAL"),
        "aspect_ratio": params.get("aspect_ratio", "source"),
        "output_quality": params.get("output_quality", "original"),
        "manual_transform": params.get("manual_transform"),
        "video_language": transcript_language or "",
        "clips": clips,
    }
    if stage:
        document["stage"] = stage
    if error:
        document["error"] = error
    return document


def build_job_response(job_id: str, status: str, manifest: list, clips: list,
                       public_domain: str, processing_time: float,
                       progress: int = 100, error: str | None = None,
                       transcript_language: str = "", params: dict | None = None) -> dict:
    """Full structured job result returned through the Vast.ai request envelope."""
    domain = (public_domain or "").rstrip("/")
    response: dict[str, Any] = {
        "job_id": job_id,
        "status": status,  # completed | failed
        "progress": progress,
        "clip_count": len(manifest),
        "download_url": f"{domain}/outputs/{job_id}.mp4",
        "manifest_url": f"{domain}/outputs/{job_id}/manifest.json",
        "clips": clips,
        "processing_time": round(processing_time, 1),
        "error": error,
        "video_language": transcript_language,
        "subtitle_style": (params or {}).get("subtitle_style", "VIRAL"),
        "aspect_ratio": (params or {}).get("aspect_ratio", "source"),
        "output_quality": (params or {}).get("output_quality", "original"),
        "manual_transform": (params or {}).get("manual_transform"),
        "finished_at": int(time.time()),
    }
    return response
