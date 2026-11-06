"""Vision engine: representative keyframe extraction (port of PY_FRAMES).

One bounded decode pass produces the coarse sampling grid (~1 frame / 8 s,
capped at 400 samples); those frames serve BOTH the AI vision input and the
coarse face scan. Precise scene cuts come from processors.scene_detection.
The chosen AI frame set (scene frames + evenly spaced interval frames,
de-duplicated, capped at 12 = the GPT-6 Luna vision budget) is cached with the
source hash, with the cache-hit path re-validating that every referenced file
still exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from cache import cache_manager
from config import VISION_CACHE_VER
from processors import scene_detection
from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("vision")

MAX_COARSE_SAMPLES = 400
MAX_AI_FRAMES = 12


def extract_keyframes(source: dict, workspace, settings) -> dict:
    """Returns the frames manifest {coarse_interval, coarse_count, scene_cuts,
    ai_frames}; reuses the computation cache when valid."""
    cache_paths = cache_manager.CachePaths(settings.cache_dir, source["source_sha256"])
    manifest_path = cache_paths.frames_manifest()

    cached = cache_manager.load_cached(
        manifest_path,
        expect={"source_sha256": source.get("source_sha256"), "pipeline": VISION_CACHE_VER},
        ttl_days=30)
    if cached is not None:
        ai_frames = cached.get("ai_frames") or []
        files_ok = all(Path(f["file"]).is_file() for f in ai_frames)
        if ai_frames and files_ok:
            logger.info("CACHE HIT: %d keyframes + scene cuts reused for source sha256:%s "
                        "(no frame decode passes this run).", len(ai_frames),
                        source["source_sha256"][:12])
            return cached
        logger.warning("cached frame manifest references missing files; re-extracting.")

    duration = source["duration"]
    coarse_dir = Path(cache_paths.coarse_dir())
    frames_dir = Path(cache_paths.frames_dir())
    coarse_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    interval = max(8.0, duration / MAX_COARSE_SAMPLES)
    cmd = [
        ff.ffmpeg_bin(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(workspace.source),
        "-an", "-vf", f"fps=1/{interval:.3f},scale=768:-2:flags=bicubic",
        "-q:v", "5", str(coarse_dir / "coarse_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"coarse frame extraction failed: {proc.stderr[-1500:]}")

    coarse_frames = []
    for index, name in enumerate(sorted(p.name for p in coarse_dir.glob("*.jpg"))):
        coarse_frames.append({"file": str(coarse_dir / name), "t": round(index * interval, 2)})
    logger.info("Extracted %d coarse frames at 1/%.1fs sampling.", len(coarse_frames), interval)

    scene_times = scene_detection.detect_scene_cuts(workspace.source, duration)
    scene_frames = scene_detection.extract_scene_frames(workspace.source, scene_times, frames_dir)

    # Choose the final AI frame set: scene-change frames + evenly spaced
    # interval frames, de-duplicated, capped at 12 (GPT-6 Luna vision budget).
    chosen = []
    used_times = []
    for frame in scene_frames:
        if any(abs(frame["t"] - t) < 2.0 for t in used_times):
            continue
        chosen.append(frame)
        used_times.append(frame["t"])
        if len(chosen) >= 6:
            break

    if coarse_frames:
        remaining = MAX_AI_FRAMES - len(chosen)
        if remaining > 0:
            step = max(1, len(coarse_frames) // remaining)
            for frame in coarse_frames[::step]:
                if len(chosen) >= MAX_AI_FRAMES:
                    break
                if any(abs(frame["t"] - t) < 2.0 for t in used_times):
                    continue
                chosen.append(frame)
                used_times.append(frame["t"])

    chosen.sort(key=lambda frame: frame["t"])
    if chosen and coarse_frames:
        if (chosen[0]["t"] > interval * 1.5 and len(chosen) < MAX_AI_FRAMES
                and not any(abs(coarse_frames[0]["t"] - t) < 2.0 for t in used_times)):
            chosen.append(coarse_frames[0])
            used_times.append(coarse_frames[0]["t"])
        if (chosen[-1]["t"] < duration - interval * 1.5 and len(chosen) < MAX_AI_FRAMES
                and not any(abs(coarse_frames[-1]["t"] - t) < 2.0 for t in used_times)):
            chosen.append(coarse_frames[-1])
            used_times.append(coarse_frames[-1]["t"])
        chosen.sort(key=lambda frame: frame["t"])

    frames_manifest = {
        "coarse_interval": round(interval, 3),
        "coarse_count": len(coarse_frames),
        "scene_cuts": scene_times,
        "ai_frames": chosen,
    }
    cache_manager.save_cached(
        manifest_path, frames_manifest,
        meta={"source_sha256": source.get("source_sha256"), "pipeline": VISION_CACHE_VER})
    logger.info("GPT-6 Luna will receive %d representative frames at %s",
                len(chosen), [f["t"] for f in chosen])
    return frames_manifest
