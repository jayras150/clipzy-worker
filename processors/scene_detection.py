"""Precise scene-change detection (FFmpeg select/showinfo pass).

Extracted from the workflow's keyframe step so scene detection can be tested
independently and reused by the vision engine.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("scene_detection")

SCENE_THRESHOLD = 0.30
MAX_SCENE_CUTS = 40


def detect_scene_cuts(source_path, duration: float) -> list[float]:
    """Scene-change timestamps via select='gt(scene,0.30)'. Sources longer than
    30 minutes skip this pass (same bounded-decode policy as the workflow)."""
    if duration <= 0 or duration > 1800:
        if duration > 1800:
            logger.info("Source longer than 30 minutes: precise scene detection skipped.")
        return []
    proc = subprocess.run(
        [ff.ffmpeg_bin(), "-nostdin", "-hide_banner", "-i", str(source_path), "-an",
         "-vf", "select='gt(scene,0.30)',showinfo", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    times = [float(m) for m in re.findall(r"pts_time:(\d+(?:\.\d+)?)", proc.stderr or "")]
    times = [t for t in times if 0.5 < t < duration - 0.5]
    if len(times) > MAX_SCENE_CUTS:
        step = len(times) / float(MAX_SCENE_CUTS)
        times = [times[int(i * step)] for i in range(MAX_SCENE_CUTS)]
    logger.info("Scene-change detection found %d cuts.", len(times))
    return times


def extract_scene_frames(source_path, scene_times: list[float], frames_dir: Path,
                          limit: int = 8) -> list[dict]:
    """Scene-cut keyframes at 768px, JPEG q5 (workflow parameters)."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for position, t in enumerate(scene_times[:limit]):
        path = frames_dir / f"scene_{position:03d}.jpg"
        proc = ff.run([
            ff.ffmpeg_bin(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(source_path), "-frames:v", "1",
            "-vf", "scale=768:-2:flags=bicubic", "-q:v", "5", str(path),
        ])
        if proc.returncode == 0 and path.exists():
            frames.append({"file": str(path), "t": round(t, 2)})
    return frames
