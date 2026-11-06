"""FFmpeg / FFprobe helpers shared by every processor.

Responsibilities:
* resolve the binaries once (ffmpeg / ffprobe from PATH or env overrides);
* verify libass `subtitles` filter support (same check as the workflow);
* probe media as JSON;
* pick the H.264 encoder: NVENC (h264_nvenc) when available and enabled,
  otherwise the workflow's libx264 CRF 18 baseline. NVENC never replaces the
  output contract: the file stays H.264 / yuv420p / AAC, and any NVENC failure
  falls back to libx264 mid-render.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache

from .logging_setup import get_logger

logger = get_logger("ffmpeg")


def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_bin() -> str:
    return os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess (list argv, never a shell string) and capture output."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _filters() -> str:
    proc = run([ffmpeg_bin(), "-hide_banner", "-filters"])
    return proc.stdout or ""


@lru_cache(maxsize=1)
def has_subtitles_filter() -> bool:
    return " subtitles " in _filters()


@lru_cache(maxsize=1)
def has_nvenc() -> bool:
    """True when this ffmpeg build can encode h264_nvenc AND an NVIDIA GPU is
    reachable. Encoders list alone is not enough (builds ship nvenc regardless
    of driver presence)."""
    proc = run([ffmpeg_bin(), "-hide_banner", "-encoders"])
    if "h264_nvenc" not in (proc.stdout or ""):
        return False
    probe = run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=black:size=64x64:rate=10:duration=0.2",
                 "-c:v", "h264_nvenc", "-f", "null", "-"])
    ok = probe.returncode == 0
    logger.info("NVENC availability check: %s", "available" if ok else "not available (CPU fallback)")
    return ok


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    proc = run([ffmpeg_bin(), "-version"])
    return (proc.stdout or "").splitlines()[0] if proc.stdout else "unknown"


def choose_encoder(allow_nvenc: bool) -> dict:
    """Return encoder settings for the render stage.

    NVENC uses the quality-driven VBR path (-rc vbr -cq 18 -b:v 0), the closest
    perceptual equivalent of libx264 CRF 18; both keep yuv420p H.264 output.
    The GOP is supplied by the caller (source-fps based, as in the workflow).
    """
    if allow_nvenc and has_nvenc():
        return {
            "codec": "h264_nvenc",
            "preset_flag": "-preset",
            "preset": "p5",
            "quality_flags": ["-rc", "vbr", "-cq", "18", "-b:v", "0", "-spatial-aq", "1"],
            "gpu": True,
        }
    return {
        "codec": "libx264",
        "preset_flag": "-preset",
        "preset": "medium",  # caller overrides to "faster" for >1920p canvases
        "quality_flags": ["-crf", "18"],
        "gpu": False,
    }


def probe_media(path: str | os.PathLike) -> dict:
    """FFprobe a media file; returns {} on failure (callers handle)."""
    proc = run([ffprobe_bin(), "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)])
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return {}


def parse_rate(value: str | None) -> float:
    try:
        if value and "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
