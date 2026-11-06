"""Android manifest builder (port of PY_MANIFEST)."""

from __future__ import annotations

import json

from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("manifest")


def build_manifest(clips_final: list, subtitles: dict, probes: dict, transcript: dict,
                   params: dict, tracks: dict, workspace) -> list:
    manifest = []
    for clip in clips_final:
        index = clip["clip_index"]
        sub = next((s for s in subtitles.get("clips", []) if s["clip_index"] == index), None)
        probe = probes.get(index, {}).get("probe", {})
        size = probes.get(index, {}).get("size_bytes", 0)
        video = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
        audio = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), {})

        visual = clip.get("visual_analysis") or {}
        manifest.append({
            "clip_index": index,
            "start_sec": clip["start_sec"],
            "end_sec": clip["end_sec"],
            "duration": round(clip["end_sec"] - clip["start_sec"], 2),
            "viral_score": clip.get("viral_score", 0),
            "topic": clip.get("topic", ""),
            "hook": clip.get("hook", ""),
            "title": clip.get("title", ""),
            "description": clip.get("description", ""),
            "tags": clip.get("tags", []),
            "language": transcript.get("language", ""),
            "visual_analysis": {
                "main_subject": visual.get("main_subject", ""),
                "scene": visual.get("scene", ""),
                "faces_detected": bool((tracks.get(index) or {}).get("primary")),
            },
            "editing": clip.get("editing", {}),
            "subtitles": {
                "burned_in": bool(sub and sub.get("burned_in")),
                "style": params["subtitle_style"],
                "language": subtitles.get("language", ""),
                "word_timing_source": subtitles.get("word_source", ""),
            },
            "subtitle_segments": (sub or {}).get("cues", []),
            "render": {
                "file": f"clip_{index}.mp4",
                "width": int(video.get("width") or 0),
                "height": int(video.get("height") or 0),
                "fps": round(ff.parse_rate(video.get("r_frame_rate")), 3),
                "video_codec": video.get("codec_name", ""),
                "audio_codec": audio.get("codec_name") or "none",
                "size_bytes": size,
            },
        })

    workspace.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifest written: %d clip entries (Android-compatible array).", len(manifest))
    for clip in manifest:
        logger.info("  clip %d: '%s' zoom=%s subs=%s manual_scale=%s",
                    clip["clip_index"], clip["title"][:48],
                    "yes" if clip["editing"].get("zoom_enabled") else "no",
                    "burned" if clip["subtitles"]["burned_in"] else "no",
                    (clip["editing"].get("manual_transform") or {}).get("scale", 1.0))
    return manifest
