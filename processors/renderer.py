"""Render plan construction + execution (port of PY_PLAN + PY_RENDER).

The plan contains, per clip: the exact ffmpeg argv, expected duration, output
geometry, the QC reference graphs, zoom windows and the full render report
(source/output resolution, fps, codecs, bitrate, crops, filter graph).
Execution runs each argv, fails loudly on any error, records libass font
warnings, and probes every output for the QC + manifest stages.
"""

from __future__ import annotations

import json
import subprocess

from processors import video_cutter
from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("renderer")


def build_render_plan(source: dict, render_hints: dict, clips_final: list,
                      subtitles: dict, params: dict, workspace, settings,
                      watermark_file) -> dict:
    geometry = render_hints["geometry"]
    encoder = ff.choose_encoder(settings.allow_nvenc)
    if encoder.get("gpu"):
        logger.info("Render encoder: h264_nvenc (GPU) with VBR cq=18; CPU libx264 remains "
                    "the automatic fallback on any NVENC failure.")
    else:
        logger.info("Render encoder: libx264 (CPU) CRF 18 preset per output size.")

    plan_clips = []
    for clip in clips_final:
        index = clip["clip_index"]
        start = clip["start_sec"]
        length = round(clip["end_sec"] - clip["start_sec"], 3)
        clip_hints = render_hints["clips"].get(str(index)) or {}

        argv = video_cutter.build_render_argv(
            clip, clip_hints, geometry, subtitles, params, source,
            workspace, str(settings.fonts_dir), watermark_file, encoder)
        graphs = video_cutter.build_graph_variants(
            clip, clip_hints, geometry, subtitles, params, workspace,
            str(settings.fonts_dir), watermark_file)

        sub = next((s for s in subtitles.get("clips", []) if s["clip_index"] == index), None)
        burned_in = bool(sub and sub.get("burned_in"))

        out_w, out_h = geometry["out_w"], geometry["out_h"]
        aw, ah = geometry["aspect_window"]["w"], geometry["aspect_window"]["h"]
        manual_window = geometry.get("manual_window") or {
            "w": geometry["src_w"], "h": geometry["src_h"], "x": 0, "y": 0}
        mw, mh = manual_window["w"], manual_window["h"]
        mx, my = manual_window["x"], manual_window["y"]
        manual_transform = geometry.get("manual_transform")
        manual_scale = float((manual_transform or {}).get("scale", 1.0))
        encode_preset = "medium" if max(out_w, out_h) <= 1920 else "faster"

        windows = clip_hints.get("zoom_windows") or []
        window_notes = "; ".join(
            f"{w['start_rel_sec']:.1f}-{w['end_rel_sec']:.1f}s @{w['zoom_level']}x "
            f"({w.get('steering', 'center')})" for w in windows) or "none"
        face_tracks = bool(clip_hints.get("crop_keypoints") or
                           any(w.get("steering", "").startswith("face-follow") for w in windows))

        if geometry["mode"] == "pad":
            final_scale = f"pad canvas {out_w}x{out_h} (blur background, fg fitted)"
        else:
            current_w, current_h = (aw, ah) if (aw, ah) != (mw, mh) else (mw, mh)
            final_scale = (f"{current_w}x{current_h} -> {out_w}x{out_h} (lanczos)"
                           if (current_w, current_h) != (out_w, out_h)
                           else "none (already at canvas size)")

        report = {
            "source_resolution": f"{geometry['src_w']}x{geometry['src_h']}",
            "source_fps": f"{source['fps']:g}",
            "source_codec": f"{source['vcodec']} ({source['pix_fmt']})",
            "source_bitrate": f"{source.get('video_bitrate', 0) // 1000} kbps",
            "output_resolution": f"{out_w}x{out_h}",
            "output_fps": f"{source['fps']:g} (source frame rate preserved)",
            "output_codec": (f"h264 (yuv420p, {encoder['codec']} "
                             + (f"NVENC VBR cq=18 preset {encode_preset}" if encoder.get("gpu")
                                else f"CRF 18 preset {encode_preset}") + ")"),
            "output_quality": geometry.get("quality_note", ""),
            "ai_zoom_intents": len(clip.get("editing", {}).get("zoom_cuts") or []),
            "validated_zoom_windows": window_notes,
            "manual_zoom_scale": f"{manual_scale:g}",
            "manual_zoom_offset": (f"({(manual_transform or {}).get('offset_x', 0)}, "
                                   f"{(manual_transform or {}).get('offset_y', 0)})"),
            "face_tracking_status": ("primary face tracked (steering zoom/aspect crop)"
                                     if face_tracks else "no primary face track (center-anchored)"),
            "final_crop": (f"manual {mw}x{mh}@({mx},{my})"
                           + (f"; aspect {aw}x{ah}" if (aw, ah) != (mw, mh) else "")
                           + (f"; zoom peak {max([w['zoom_level'] for w in windows] or [1.0]):g}x"
                              if windows else "")),
            "final_scale": final_scale,
            "filter_graph": graphs["video_graph"],
        }

        filter_paths = []
        if burned_in:
            filter_paths.append(sub.get("file") or str(workspace.work / "subs" / f"clip_{index}.ass"))
        filter_paths.append(str(settings.fonts_dir))
        if watermark_file:
            filter_paths.append(str(watermark_file))
        filter_cwd = video_cutter.filter_working_directory(*filter_paths)

        plan_clips.append({
            "clip_index": index,
            "cwd": filter_cwd,
            "argv": argv,
            "output": str(workspace.output_clips / f"clip_{index}.mp4"),
            "expect_duration": length,
            "out_w": out_w, "out_h": out_h,
            "fps": source["fps"],
            "has_audio": source["has_audio"],
            "burned_in": burned_in,
            "ass": (sub.get("file") if burned_in else None),
            "video_graph": graphs["video_graph"],
            "qc_graph": graphs["qc_graph"],
            "qc_nozoom_graph": graphs["qc_nozoom_graph"],
            "clip_start": start,
            "zoom_windows": windows,
            "encode_preset": encode_preset,
            "encoder": encoder["codec"],
            "report": report,
        })
        zoom_note = f", {len(windows)} zoom window(s)" if windows else ", no zoom"
        logger.info("Planned clip %d: %dx%d @ %g fps, %s %s%s, manual scale %g, subtitles=%s.",
                    index, out_w, out_h, source["fps"], encoder["codec"],
                    "NVENC cq18" if encoder.get("gpu") else "CRF 18", zoom_note,
                    manual_scale, "burned" if burned_in else "none")

    workspace.output_clips.mkdir(parents=True, exist_ok=True)
    plan = {"clips": plan_clips, "geometry": geometry, "encoder": encoder}
    (workspace.work / "render_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    logger.info("Render plan ready for %d clip(s), mode=%s, canvas %dx%d (%s).",
                len(plan_clips), geometry["mode"], geometry["out_w"], geometry["out_h"],
                geometry.get("output_quality", "original"))
    return plan


def _render_one_with_fallback(entry: dict) -> None:
    """Run the planned argv; on failure with NVENC, retry once with libx264."""
    proc = subprocess.run(entry["argv"], capture_output=True, text=True, cwd=entry.get("cwd"))
    if proc.returncode != 0 and entry.get("encoder") == "h264_nvenc":
        logger.warning("NVENC render failed for clip %s; retrying once with libx264 (CPU "
                       "fallback). Error tail: %s", entry["clip_index"],
                       "\n".join((proc.stderr or "").strip().splitlines()[-8:]))
        cpu_argv = list(entry["argv"])
        try:
            codec_index = cpu_argv.index("-c:v") + 1
        except ValueError:  # pragma: no cover - argv always contains -c:v
            raise RuntimeError(f"ffmpeg failed for clip {entry['clip_index']}:\n"
                               f"{(proc.stderr or '')[-2000:]}")
        preset_index = cpu_argv.index("-preset", codec_index) + 1
        # Replace codec + preset + NVENC quality flags with the CPU baseline.
        cpu_argv[codec_index] = "libx264"
        cpu_argv[preset_index] = entry.get("encode_preset", "medium")
        # Drop NVENC-only quality flags (-rc vbr -cq N -b:v 0 -spatial-aq 1).
        cleaned = []
        skip = 0
        for i, arg in enumerate(cpu_argv):
            if skip > 0:
                skip -= 1
                continue
            if i in (codec_index, preset_index):
                cleaned.append(arg)
                continue
            if i == preset_index - 1:
                cleaned.append(arg)
                continue
            if arg in ("-rc", "-cq", "-b:v", "-spatial-aq") and i > preset_index:
                skip = 1 if arg != "-b:v" else 0  # -b:v 0 spans two tokens
                if arg == "-b:v":
                    skip = 1
                continue
            cleaned.append(arg)
        # Insert CRF 18 before -g.
        g_index = cleaned.index("-g")
        cleaned[g_index:g_index] = ["-crf", "18"]
        proc = subprocess.run(cleaned, capture_output=True, text=True, cwd=entry.get("cwd"))
        if proc.returncode == 0:
            entry["encoder"] = "libx264"
            entry["report"]["output_codec"] = "h264 (yuv420p, libx264 CRF 18 - NVENC fallback)"
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for clip {entry['clip_index']}:\n"
            + "\n".join((proc.stderr or "").strip().splitlines()[-40:]))
    entry["_stderr"] = proc.stderr or ""


def render_clips(plan: dict, workspace) -> list[dict]:
    """Execute the plan; returns the outputs_probe list for QC + manifest."""
    import os

    probes = []
    for entry in plan["clips"]:
        logger.info("Rendering clip %s (%ss) -> %s", entry["clip_index"],
                    entry["expect_duration"], entry["output"])
        _render_one_with_fallback(entry)
        stderr = entry.pop("_stderr", "")
        for line in stderr.splitlines():
            if "fontselect" in line or "libass" in line.lower() or "glyf" in line.lower():
                logger.info("  libass: %s", line.strip()[:180])
        output = entry["output"]
        if not os.path.exists(output) or os.path.getsize(output) < 10240:
            raise RuntimeError(f"clip {entry['clip_index']} output is missing or "
                               "suspiciously small.")
        info = ff.probe_media(output)
        probes.append({"clip_index": entry["clip_index"], "probe": info,
                       "size_bytes": os.path.getsize(output)})
        logger.info("  done: %.2f MB", os.path.getsize(output) / 1e6)

        # ----- Render report (full media pipeline visibility) ----------
        out_bitrate = 0
        try:
            out_bitrate = int(info.get("format", {}).get("bit_rate") or 0)
        except (TypeError, ValueError):
            out_bitrate = 0
        rep = entry["report"]
        logger.info("=" * 66)
        logger.info("RENDER REPORT - clip %s", entry["clip_index"])
        for key, value in (
                ("SOURCE RESOLUTION", rep["source_resolution"]),
                ("SOURCE FPS", rep["source_fps"]),
                ("SOURCE CODEC", rep["source_codec"]),
                ("SOURCE BITRATE", rep["source_bitrate"]),
                ("OUTPUT RESOLUTION", rep["output_resolution"]),
                ("OUTPUT FPS", rep["output_fps"]),
                ("OUTPUT CODEC", rep["output_codec"]),
                ("OUTPUT BITRATE", f"{out_bitrate // 1000} kbps"),
                ("OUTPUT FILE SIZE", f"{os.path.getsize(output) / 1e6:.2f} MB"),
                ("OUTPUT QUALITY POLICY", rep["output_quality"]),
                ("AI ZOOM INTENTS", rep["ai_zoom_intents"]),
                ("VALIDATED ZOOM WINDOWS", rep["validated_zoom_windows"]),
                ("MANUAL ZOOM SCALE", rep["manual_zoom_scale"]),
                ("MANUAL ZOOM OFFSET", rep["manual_zoom_offset"]),
                ("FACE TRACKING STATUS", rep["face_tracking_status"]),
                ("FINAL CROP", rep["final_crop"]),
                ("FINAL SCALE", rep["final_scale"]),
                ("FFMPEG FILTER GRAPH", rep["filter_graph"])):
            logger.info("%s: %s", key, value)
        logger.info("=" * 66)

    (workspace.work / "outputs_probe.json").write_text(
        json.dumps(probes, indent=2), encoding="utf-8")
    logger.info("Rendered %d clip(s).", len(plan["clips"]))
    return probes
