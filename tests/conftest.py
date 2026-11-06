"""Shared test fixtures: sample videos, settings, canned AI responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import WorkerSettings  # noqa: E402


def _make_sample_video(path: Path, width: int = 640, height: int = 360,
                       duration: float = 24.0, fps: float = 30.0,
                       with_audio: bool = True) -> None:
    """Synthetic test source: animated testsrc2 pattern + sine audio."""
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "raw_video.mp4"
    _make_sample_video(path)
    return path


@pytest.fixture()
def settings(tmp_path, monkeypatch) -> WorkerSettings:
    """Hermetic settings: temp dirs, no R2 requirement, no real API keys."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENROUTER_CHAT_MODEL", "openai/gpt-6-luna")
    monkeypatch.setenv("OPENROUTER_STT_MODEL", "openai/whisper-large-v3-turbo")
    monkeypatch.setenv("CLIPZY_WORK_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("CLIPZY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLIPZY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CLIPZY_FONTS_DIR", str(tmp_path / "fonts"))
    monkeypatch.setenv("CLIPZY_PIPELINE_LOG_FILE", str(tmp_path / "pipeline.log"))
    monkeypatch.setenv("CLIPZY_STATUS_UPLOADS", "false")
    monkeypatch.setenv("CLIPZY_R2_REQUIRED", "false")
    monkeypatch.setenv("CLIPZY_ALLOW_NVENC", "false")
    s = WorkerSettings()
    s.ensure_dirs()
    return s


# ---------------------------------------------------------------------------
# Canned AI responses (never any network)
# ---------------------------------------------------------------------------
CANNED_TRANSCRIPT_SEGMENTS = [
    {"start": 1.0, "end": 4.2, "text": "Welcome to the channel everyone"},
    {"start": 4.5, "end": 8.0, "text": "Today we are talking about something surprising"},
    {"start": 8.5, "end": 12.0, "text": "The numbers are absolutely incredible"},
    {"start": 12.5, "end": 16.0, "text": "Here is the part nobody expects"},
    {"start": 16.5, "end": 20.0, "text": "And that is why this matters so much"},
]


def _expand(text):
    """(word, rel_start, rel_end) tuples spread evenly across a segment."""
    out = []
    cursor = 0.0
    for word in text.split():
        out.append((word, cursor, cursor + 0.35))
        cursor += 0.4
    return out


CANNED_WORDS = []
for seg in CANNED_TRANSCRIPT_SEGMENTS:
    for word, ws, we in _expand(seg["text"]):
        CANNED_WORDS.append({"word": word,
                             "start": round(seg["start"] + ws, 3),
                             "end": round(seg["start"] + we, 3)})


def canned_transcript_payload(offset: float = 0.0) -> dict:
    """verbose_json-shaped payload used to fake the OpenRouter STT endpoint."""
    segments = []
    for seg in CANNED_TRANSCRIPT_SEGMENTS:
        words = [{"word": w["word"], "start": w["start"] - offset,
                  "end": w["end"] - offset}
                 for w in CANNED_WORDS
                 if seg["start"] <= w["start"] < seg["end"]]
        segments.append({"start": seg["start"] - offset, "end": seg["end"] - offset,
                         "text": seg["text"], "words": words})
    return {"language": "en", "segments": segments}


CANNED_CURATION = {
    "video_analysis": {
        "topic": "Synthetic test video",
        "summary": "A generated test pattern with narration.",
        "main_subjects": ["the narrator"],
        "scenes": ["test pattern"],
        "language": "en",
    },
    "clips": [
        {
            "clip_index": 1,
            "start_sec": 1.0,
            "end_sec": 12.0,
            "viral_score": 88,
            "topic": "surprising numbers",
            "hook": "Today we are talking about something surprising",
            "title": "The Numbers Nobody Believes",
            "description": "The narrator reveals incredible numbers that nobody expects.",
            "tags": ["#testing", "#numbers", "#surprise", "#synthetic"],
            "visual_analysis": {"main_subject": "the narrator", "scene": "test pattern"},
            "zoom_intents": [
                {"start_rel_sec": 1.0, "end_rel_sec": 5.0, "zoom_level": 1.10,
                 "target": "primary_subject", "reason": "key emphasis moment"}
            ],
        },
        {
            "clip_index": 2,
            "start_sec": 12.5,
            "end_sec": 22.0,
            "viral_score": 72,
            "topic": "the unexpected part",
            "hook": "",
            "title": "The Part Nobody Expects",
            "description": "The final revelation and why it matters.",
            "tags": ["#testing", "#revelation", "#finale", "#synthetic"],
            "visual_analysis": {"main_subject": "the narrator", "scene": "test pattern"},
            "zoom_intents": [],
        },
    ],
}

CANNED_REFINE = {
    "clips": [
        {"clip_index": 1, "topic": "surprising numbers",
         "title": "The Numbers Nobody Believes (Refined)",
         "description": "Refined: the narrator reveals incredible numbers.",
         "tags": ["#testing", "#numbers", "#surprise", "#synthetic"],
         "hook": "Today we are talking about something surprising",
         "visual_analysis": {"main_subject": "the narrator", "scene": "test pattern"}},
        {"clip_index": 2, "topic": "the unexpected part",
         "title": "The Part Nobody Expects (Refined)",
         "description": "Refined: the final revelation.",
         "tags": ["#testing", "#revelation", "#finale", "#synthetic"],
         "hook": "",
         "visual_analysis": {"main_subject": "the narrator", "scene": "test pattern"}},
    ]
}


def fake_chat_info(content: str, model: str = "openai/gpt-6-luna") -> dict:
    return {"ok": True, "http_status": 200, "finish_reason": "stop",
            "content": content, "usage": {"prompt_tokens": 100,
                                          "completion_tokens": 200,
                                          "total_tokens": 300},
            "served_model": model, "error": ""}


@pytest.fixture()
def fake_gpt(monkeypatch):
    """Patch gpt_client.chat_with_retries to serve canned curation/refine JSON."""
    from engines import gpt_client

    calls = []

    def fake_chat(api_base, api_key, payload, timeout=300, transport_retries=4, log=print):
        calls.append(payload)
        messages = payload.get("messages", [])
        is_multimodal = any(
            isinstance(m.get("content"), list) for m in messages)
        content = json.dumps(CANNED_CURATION if is_multimodal else CANNED_REFINE,
                             ensure_ascii=False)
        return fake_chat_info(content, payload.get("model", "openai/gpt-6-luna"))

    monkeypatch.setattr(gpt_client, "chat_with_retries", fake_chat)
    return calls


@pytest.fixture()
def fake_stt(monkeypatch):
    """Patch the OpenRouter STT call to serve the canned transcript."""
    from engines import transcription

    calls = []

    def fake_remote(chunk, api_base, api_key, stt_model, language_hint, log=print):
        calls.append(chunk)
        return canned_transcript_payload(offset=chunk["start"])

    monkeypatch.setattr(transcription, "_transcribe_chunk_remote", fake_remote)
    return calls


class FakeR2:
    """Records uploads without touching the network (stores raw bodies)."""

    def __init__(self, public_domain: str = "https://cdn.example.com") -> None:
        self.public_domain = public_domain.rstrip("/")
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []

    def _put(self, key: str, body: bytes, content_type: str) -> None:
        self.calls.append((key, content_type))
        self.objects[key] = body

    def upload_file(self, local_path, key, content_type=None) -> str:
        data = Path(local_path).read_bytes()
        self._put(key, data, content_type or "application/octet-stream")
        return self.public_url(key)

    def upload_json(self, document, key) -> str:
        self._put(key, json.dumps(document).encode("utf-8"), "application/json")
        return self.public_url(key)

    def upload_directory(self, local_dir, key_prefix, content_type="application/octet-stream"):
        urls = []
        for path in sorted(Path(local_dir).iterdir()):
            if path.is_file():
                urls.append(self.upload_file(path, f"{key_prefix}/{path.name}", content_type))
        return urls

    def public_url(self, key) -> str:
        return f"{self.public_domain}/{key.lstrip('/')}"

    def upload_job_outputs(self, job_id, clips_dir, manifest_document) -> dict:
        clip_files = sorted(Path(clips_dir).glob("clip_*.mp4"))
        self.upload_directory(clips_dir, f"outputs/{job_id}", "video/mp4")
        self.upload_file(clip_files[0], f"outputs/{job_id}.mp4", "video/mp4")
        self.upload_json(manifest_document, f"outputs/{job_id}_meta.json")
        self.upload_json(manifest_document, f"outputs/{job_id}/manifest.json")
        return {"primary": self.public_url(f"outputs/{job_id}.mp4"),
                "manifest": self.public_url(f"outputs/{job_id}/manifest.json")}

    def upload_status(self, job_id, status_document) -> str:
        return self.upload_json(status_document, f"status/{job_id}.json")

    def upload_debug(self, job_id, debug_dir, extra_files) -> list:
        return []


@pytest.fixture()
def fake_r2() -> FakeR2:
    return FakeR2()
