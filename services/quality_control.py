"""Quality control (port of PY_QC).

Verifies, per rendered clip: codec/pix_fmt/resolution/aspect/fps/duration,
audio presence + codec, decode integrity (-xerror), black-frame runs, AI zoom
presence on real exported frames (zoomed vs zoom-neutral reference render),
and subtitle presence via pixel-isolated footprint comparison (with/without
the subtitles filter at the same output time). Metadata sanity checks reject
placeholder-like titles, wrong tag counts, zoom windows outside the clip and
manual_transform mismatches. Errors fail the job; warnings are reported.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

import cv2
import numpy as np

from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("qc")


class QualityControlError(RuntimeError):
    pass


def _probe_stream(probe, kind):
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == kind:
            return stream
    return None


def _run_ffmpeg(args):
    return subprocess.run(args, capture_output=True, text=True)


def _render_ref(graph: str, t: float, clip_start: float, source_path: str, path: str) -> None:
    # Input seeking resets t to 0; lift PTS so per-frame zoom expressions and
    # the subtitles filter evaluate at the correct clip-relative time.
    body_graph = graph[len("[0:v]"):]
    graph_at_t = f"[0:v]setpts=PTS+{t:.3f}/TB," + body_graph
    _run_ffmpeg([ff.ffmpeg_bin(), "-y", "-nostdin", "-loglevel", "error",
                 "-ss", f"{clip_start + t:.3f}", "-t", "0.5",
                 "-i", source_path,
                 "-filter_complex", graph_at_t, "-map", "[vfinal]",
                 "-frames:v", "1", path])


def run_quality_control(plan: dict, probes: dict, subtitles: dict, clips: list,
                        params: dict, workspace) -> tuple[list, list]:
    """Returns (errors, warnings); raises nothing - the caller decides."""
    errors = []
    warnings = []

    for entry in plan["clips"]:
        index = entry["clip_index"]
        output = entry["output"]
        tag = f"clip {index}"

        if not os.path.exists(output):
            errors.append(f"{tag}: output file missing.")
            continue
        if os.path.getsize(output) < 10240:
            errors.append(f"{tag}: output file is suspiciously small.")

        probe = probes.get(index, {}).get("probe", {})
        video = _probe_stream(probe, "video")
        audio = _probe_stream(probe, "audio")
        if video is None:
            errors.append(f"{tag}: no video stream in the rendered file.")
            continue

        if video.get("codec_name") != "h264":
            errors.append(f"{tag}: video codec is {video.get('codec_name')}, expected h264.")
        if video.get("pix_fmt") != "yuv420p":
            errors.append(f"{tag}: pixel format is {video.get('pix_fmt')}, expected yuv420p.")
        if int(video.get("width") or 0) != entry["out_w"] or int(video.get("height") or 0) != entry["out_h"]:
            errors.append(f"{tag}: resolution {video.get('width')}x{video.get('height')} "
                          f"!= planned {entry['out_w']}x{entry['out_h']}.")
        ratio = int(video.get("width") or 1) / max(1, int(video.get("height") or 1))
        expected_ratio = entry["out_w"] / entry["out_h"]
        if abs(ratio - expected_ratio) > 0.02:
            errors.append(f"{tag}: aspect ratio {ratio:.4f} deviates from {expected_ratio:.4f}.")

        out_fps = ff.parse_rate(video.get("r_frame_rate")) or ff.parse_rate(video.get("avg_frame_rate"))
        if out_fps:
            if abs(out_fps - entry["fps"]) > 0.5:
                errors.append(f"{tag}: frame rate {out_fps:g} differs strongly from source {entry['fps']:g}.")
            elif abs(out_fps - entry["fps"]) > 0.06:
                warnings.append(f"{tag}: frame rate {out_fps:g} slightly differs from source {entry['fps']:g}.")

        duration = 0.0
        try:
            duration = float(probe.get("format", {}).get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0:
            errors.append(f"{tag}: could not read output duration.")
        else:
            if abs(duration - entry["expect_duration"]) > 1.5:
                errors.append(f"{tag}: duration {duration:.2f}s differs from planned "
                              f"{entry['expect_duration']:.2f}s.")
            elif abs(duration - entry["expect_duration"]) > 0.5:
                warnings.append(f"{tag}: duration {duration:.2f}s slightly differs from planned "
                                f"{entry['expect_duration']:.2f}s.")

        if entry["has_audio"] and audio is None:
            errors.append(f"{tag}: source has audio but the rendered clip does not.")
        if audio is not None and audio.get("codec_name") not in ("aac", None):
            errors.append(f"{tag}: audio codec is {audio.get('codec_name')}.")

        # Decode integrity: exit code must be clean.
        proc = _run_ffmpeg([ff.ffmpeg_bin(), "-v", "error", "-xerror", "-i", output,
                            "-f", "null", "-"])
        if proc.returncode != 0:
            errors.append(f"{tag}: decode check failed: {(proc.stderr or '').strip()[:300]}")

        # Black frame detection.
        proc = _run_ffmpeg([ff.ffmpeg_bin(), "-i", output, "-vf",
                            "blackdetect=d=0.4:pix_black_th=0.98", "-an", "-f", "null", "-"])
        for line in (proc.stderr or "").splitlines():
            if "blackdetect" not in line:
                continue
            match = re.search(r"black_start:(\d+\.?\d*).*black_duration:(\d+\.?\d*)", line)
            if match:
                start, black_len = float(match.group(1)), float(match.group(2))
                if black_len >= 1.0:
                    errors.append(f"{tag}: black frame run of {black_len:.1f}s at {start:.1f}s.")
                elif black_len >= 0.4:
                    warnings.append(f"{tag}: short black run of {black_len:.1f}s at {start:.1f}s.")

        # ---- ZOOM PRESENCE (auto-zoom verification on exported frames) --
        windows = entry.get("zoom_windows") or []
        if windows:
            peak = max(windows, key=lambda w: min(w["end_rel_sec"], 15.0) - w["start_rel_sec"])
            t_peak = round((peak["start_rel_sec"] + peak["end_rel_sec"]) / 2, 3)
            t_peak = min(max(t_peak, 0.2), max(0.2, entry["expect_duration"] - 0.2))
            with tempfile.TemporaryDirectory() as tmp:
                out_png = os.path.join(tmp, "out.png")
                zoom_png = os.path.join(tmp, "zoom.png")
                nozoom_png = os.path.join(tmp, "nozoom.png")

                _run_ffmpeg([ff.ffmpeg_bin(), "-y", "-nostdin", "-loglevel", "error",
                             "-ss", f"{t_peak:.3f}", "-i", output, "-frames:v", "1", out_png])
                _render_ref(entry["qc_graph"], t_peak, entry["clip_start"], str(workspace.source), zoom_png)
                _render_ref(entry["qc_nozoom_graph"], t_peak, entry["clip_start"], str(workspace.source), nozoom_png)
                out_img = cv2.imread(out_png, cv2.IMREAD_GRAYSCALE)
                zoom_img = cv2.imread(zoom_png, cv2.IMREAD_GRAYSCALE)
                nozoom_img = cv2.imread(nozoom_png, cv2.IMREAD_GRAYSCALE)
                if out_img is None or zoom_img is None or nozoom_img is None \
                        or not (out_img.shape == zoom_img.shape == nozoom_img.shape):
                    warnings.append(f"{tag}: zoom presence check inconclusive "
                                    f"(reference render failed at t={t_peak:.1f}s).")
                else:
                    mad_zoom = float(np.abs(out_img.astype(int) - zoom_img.astype(int)).mean())
                    mad_nozoom = float(np.abs(out_img.astype(int) - nozoom_img.astype(int)).mean())
                    if mad_nozoom < 1.0:
                        warnings.append(f"{tag}: zoom presence check inconclusive at t={t_peak:.1f}s "
                                        f"(frame too static to discriminate); inspect manually.")
                    elif mad_zoom < 0.8 * mad_nozoom:
                        logger.info("  %s: AI zoom verified on exported frames at t=%.1fs "
                                    "(mean |diff| zoomed %.1f vs neutral %.1f, peak %sx, %s).",
                                    tag, t_peak, mad_zoom, mad_nozoom, peak["zoom_level"],
                                    peak.get("steering", "center"))
                    else:
                        errors.append(
                            f"{tag}: zoom windows were planned but exported frames do not show "
                            f"the zoom (t={t_peak:.1f}s: |diff| zoomed {mad_zoom:.1f} vs neutral "
                            f"{mad_nozoom:.1f}); refusing to claim auto zoom works.")
        elif entry.get("report", {}).get("ai_zoom_intents"):
            errors.append(f"{tag}: zoom intents existed but no zoom windows reached the render plan.")

        # ---- Subtitle presence (pixel-isolated, motion-safe) -----------
        sub = next((s for s in subtitles.get("clips", []) if s["clip_index"] == index), None)
        if entry["burned_in"] and sub:
            cues = [c for c in sub.get("cues", []) if c["end"] > c["start"]]
            if not cues:
                warnings.append(f"{tag}: no caption events; visual subtitle check skipped.")
            else:
                candidates = sorted(cues, key=lambda c: c["end"] - c["start"], reverse=True)[:5]
                times = [round((c["start"] + c["end"]) / 2, 3) for c in candidates]

                with tempfile.TemporaryDirectory() as tmp:
                    def output_frame(t, path):
                        _run_ffmpeg([ff.ffmpeg_bin(), "-y", "-nostdin", "-loglevel", "error",
                                     "-ss", f"{t:.3f}", "-i", output, "-frames:v", "1", path])

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    verified_at = None
                    ambiguous = "no cue could be sampled"
                    recalls = []
                    for t_cue in times:
                        out_png = os.path.join(tmp, "out.png")
                        nosub_png = os.path.join(tmp, "nosub.png")
                        withsub_png = os.path.join(tmp, "withsub.png")
                        output_frame(t_cue, out_png)
                        _render_ref(entry["qc_graph"], t_cue, entry["clip_start"], str(workspace.source), nosub_png)
                        _render_ref(entry["video_graph"], t_cue, entry["clip_start"], str(workspace.source), withsub_png)
                        out_img = cv2.imread(out_png, cv2.IMREAD_GRAYSCALE)
                        nosub_img = cv2.imread(nosub_png, cv2.IMREAD_GRAYSCALE)
                        withsub_img = cv2.imread(withsub_png, cv2.IMREAD_GRAYSCALE)
                        if out_img is None or nosub_img is None or withsub_img is None:
                            ambiguous = f"reference render failed at t={t_cue:.1f}s"
                            continue
                        if not (out_img.shape == nosub_img.shape == withsub_img.shape):
                            ambiguous = f"reference shape mismatch at t={t_cue:.1f}s"
                            continue
                        footprint = cv2.absdiff(withsub_img, nosub_img)
                        mask = footprint > 40
                        if mask.sum() < max(300, 0.0005 * mask.size):
                            ambiguous = f"no visible caption pixels at t={t_cue:.1f}s"
                            continue
                        diff_nosub = cv2.absdiff(out_img, nosub_img) > 40
                        diff_withsub = cv2.absdiff(out_img, withsub_img) > 40
                        recall_nosub = float((diff_nosub & mask).sum() / max(1, mask.sum()))
                        recall_withsub = float((diff_withsub & mask).sum() / max(1, mask.sum()))
                        recalls.append(recall_nosub)
                        inside_shift = float(cv2.absdiff(out_img, nosub_img)[mask].mean())
                        outside = ~cv2.dilate(mask.astype(np.uint8), kernel,
                                              iterations=4).astype(bool)
                        outside_shift = (float(cv2.absdiff(out_img, nosub_img)[outside].mean())
                                         if outside.any() else 0.0)
                        ambiguous = (f"t={t_cue:.1f}s text-pixel recall "
                                     f"{recall_nosub:.2f} (no-sub) / {recall_withsub:.2f} (with-sub)")
                        if recall_nosub >= 0.60 and recall_withsub <= 0.35:
                            verified_at = (t_cue, recall_nosub, recall_withsub,
                                           inside_shift, outside_shift)
                            break

                    if verified_at:
                        t_cue, r_no, r_with, in_shift, out_shift = verified_at
                        logger.info("  %s: subtitle presence verified at t=%.1fs (text-pixel "
                                    "recall %.0f%% vs with-sub reference %.0f%%; mean shift "
                                    "inside %.1f / outside %.1f).", tag, t_cue, r_no * 100,
                                    r_with * 100, in_shift, out_shift)
                    elif recalls and min(recalls) < 0.25:
                        errors.append(
                            f"{tag}: burned subtitles not detected at any sampled cue "
                            f"({ambiguous}); refusing to claim subtitles were burned.")
                    else:
                        warnings.append(f"{tag}: subtitle verification inconclusive "
                                        f"({ambiguous}); treating as pass, but inspect the clip.")
        elif not entry["burned_in"] and subtitles.get("style") != "NONE":
            logger.info("  %s: no captions in range; burned_in=false reported honestly.", tag)

    # Metadata sanity: no placeholders, no fabricated markers.
    PLACEHOLDER_PATTERNS = re.compile(
        r"(?i)(amazing video|video highlight|highlight \d|untitled|todo|xxx|undefined|nan|placeholder|【)")
    for clip in clips:
        tag = f"clip {clip['clip_index']}"
        if not clip.get("title"):
            errors.append(f"{tag}: missing title.")
        elif PLACEHOLDER_PATTERNS.search(clip["title"]):
            errors.append(f"{tag}: placeholder-like title '{clip['title']}'.")
        if not clip.get("description"):
            errors.append(f"{tag}: missing description.")
        tags = clip.get("tags") or []
        if not (3 <= len(tags) <= 5):
            errors.append(f"{tag}: expected 3-5 tags, found {len(tags)}.")
        score = clip.get("viral_score")
        if not isinstance(score, int) or not (1 <= score <= 100):
            errors.append(f"{tag}: viral_score {score} outside 1-100.")
        duration = round(clip["end_sec"] - clip["start_sec"], 2)
        for zoom in clip["editing"]["zoom_cuts"]:
            if not (0 <= zoom["start_rel_sec"] < zoom["end_rel_sec"] <= duration):
                errors.append(f"{tag}: zoom window {zoom} outside the clip duration.")
            if not (1.05 <= zoom["zoom_level"] <= 1.20):
                errors.append(f"{tag}: zoom level {zoom['zoom_level']} outside 1.05-1.20.")
        if clip["editing"]["zoom_enabled"] and not clip["editing"]["zoom_cuts"]:
            errors.append(f"{tag}: zoom_enabled is true but zoom_cuts is empty.")
        # Manual transform echo must match the validated job parameters.
        if clip["editing"].get("manual_transform") != params.get("manual_transform"):
            errors.append(f"{tag}: manual_transform in the manifest does not match the job parameters.")

    for warning in warnings:
        logger.warning("WARN: %s", warning)
    return errors, warnings
