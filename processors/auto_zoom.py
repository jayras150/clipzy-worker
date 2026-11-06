"""Zoom reconciliation (the clip half of the workflow's PY_ZOOM).

Validates each AI zoom intent against scene cuts, duration windows and the
resolution budget; merges near windows; re-derives smooth face-follow
keypoints from the tracked primary subject (remapped into manual-window
coordinates); computes subject-aware crop keypoints for aspect-crop mode and
the caption placement band (captions stay away from the primary face band).
"""

from __future__ import annotations

from utils.logging_setup import get_logger

logger = get_logger("zoom")


def reconcile_zooms(source: dict, curation: dict, tracks: dict, face_context: dict,
                    params: dict, geometry: dict) -> tuple[list, dict]:
    """Returns (clips_final, render_hints)."""
    src_w, src_h = geometry["src_w"], geometry["src_h"]
    aw, ah = geometry["aspect_window"]["w"], geometry["aspect_window"]["h"]
    mode = geometry["mode"]
    out_w, out_h = geometry["out_w"], geometry["out_h"]
    z_max = geometry["z_max"]
    zoom_globally_enabled = geometry["zoom_globally_enabled"]
    manual_transform = geometry["manual_transform"]
    manual_active = manual_transform is not None
    mw, mh = (geometry["manual_window"] or {"w": src_w, "h": src_h, "x": 0, "y": 0})["w"], \
             (geometry["manual_window"] or {"w": src_w, "h": src_h, "x": 0, "y": 0})["h"]
    mx, my = (geometry["manual_window"] or {"w": src_w, "h": src_h, "x": 0, "y": 0})["x"], \
             (geometry["manual_window"] or {"w": src_w, "h": src_h, "x": 0, "y": 0})["y"]
    s = geometry["manual_scale"]

    clips_final = []
    render_clips = {}
    for clip in curation["clips"]:
        index = clip["clip_index"]
        length = round(clip["end_sec"] - clip["start_sec"], 3)
        track = tracks.get(index) or {}
        primary = track.get("primary")
        center_samples = (primary or {}).get("center_samples") or []
        cuts = track.get("cuts") or []
        sample_fps = track.get("sample_fps") or 2.0
        dropped_reasons = []
        accepted = []

        if not zoom_globally_enabled:
            dropped_reasons.append("global zoom budget too low for this source resolution")

        for intent in clip.get("zoom_intents") or []:
            try:
                ws = max(0.0, min(length - 0.5, float(intent["start_rel_sec"])))
                we = min(length, float(intent["end_rel_sec"]))
            except (KeyError, TypeError, ValueError):
                dropped_reasons.append("malformed zoom window")
                continue
            if we - ws < 1.2:
                dropped_reasons.append(f"window {ws:.1f}-{we:.1f}s too short")
                continue
            inner_cuts = [c for c in cuts if ws < c < we]
            if inner_cuts:
                boundaries = [ws] + inner_cuts + [we]
                spans = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
                ws, we = max(spans, key=lambda span: span[1] - span[0])
                if we - ws < 1.2:
                    dropped_reasons.append("scene cut inside the zoom window")
                    continue
            try:
                level = min(z_max, float(intent.get("zoom_level") or 1.10))
            except (TypeError, ValueError):
                level = 1.10
            if level < 1.06:
                dropped_reasons.append("resolution budget too low for this zoom level")
                continue
            # AUTO ZOOM FIX: a tracked primary subject is OPTIONAL steering,
            # never a hard requirement. GPT-6 Luna already had to justify every
            # zoom intent visually; when face tracking fails (no faces,
            # detector unavailable, coverage gaps) the zoom is anchored at the
            # center of the manual window instead of being silently discarded.
            if primary is not None:
                in_window = [sm for sm in center_samples if ws - 0.5 <= sm["t"] <= we + 0.5]
                expected = max(1, int(round((we - ws) * sample_fps)) + 1)
                coverage = len(in_window) / expected
                steering = "face-follow"
                if coverage < 0.5:
                    steering = "center (face coverage "
                    steering += f"{coverage:.0%} below 50% inside the window)"
            else:
                steering = "center (no tracked primary subject)"
            accepted.append({"ws": round(ws, 2), "we": round(we, 2),
                             "level": round(level, 2),
                             "steering": steering,
                             "reason": intent.get("reason", "")})

        # Sort, merge windows that are too close, cap durations and count.
        accepted.sort(key=lambda z: z["ws"])
        merged = []
        for zoom in accepted:
            if merged and zoom["ws"] - merged[-1]["we"] < 1.6:
                merged[-1]["we"] = max(merged[-1]["we"], zoom["we"])
                merged[-1]["level"] = max(merged[-1]["level"], zoom["level"])
            else:
                merged.append(zoom)
        for zoom in merged:
            if zoom["we"] - zoom["ws"] > 15.0:
                zoom["we"] = round(zoom["ws"] + 15.0, 2)
        merged = merged[:3]

        # Re-derive smooth follow keypoints from the tracked center samples,
        # remapped into MANUAL-WINDOW coordinates.
        zoom_windows = []
        for zoom in merged:
            half = 0.5 / zoom["level"]
            points = []
            if zoom["steering"].startswith("face-follow"):
                for sample in center_samples:
                    if zoom["ws"] - 0.5 <= sample["t"] <= zoom["we"] + 0.5:
                        x_m = (sample["x"] * src_w - mx) / mw
                        y_m = (sample["y"] * src_h - my) / mh
                        x = min(1.0 - half, max(half, x_m))
                        y = min(1.0 - half, max(half, y_m))
                        points.append((round(sample["t"], 2), round(x, 4), round(y, 4)))
            if len(points) > 6:
                step = (len(points) - 1) / 5.0
                points = [points[int(i * step)] for i in range(6)]
            if not points:
                points = [(zoom["ws"], 0.5, 0.5)]
            zoom_windows.append({
                "start_rel_sec": zoom["ws"], "end_rel_sec": zoom["we"],
                "zoom_level": zoom["level"], "keypoints": points,
                "steering": zoom["steering"], "reason": zoom["reason"],
            })

        zoom_cuts = [{"start_rel_sec": z["start_rel_sec"], "end_rel_sec": z["end_rel_sec"],
                      "zoom_level": z["zoom_level"], "target": "primary_subject"}
                     for z in zoom_windows]

        # Subject-aware crop following for aspect-crop mode (manual-window
        # pixel coords relative to the window origin).
        crop_keypoints = None
        if (mode == "crop" and center_samples and aw <= mw and ah <= mh):
            points = []
            for sample in center_samples:
                x_px = min(max((sample["x"] * src_w - mx) - aw / 2, 0), mw - aw)
                y_px = min(max((sample["y"] * src_h - my) - ah / 2, 0), mh - ah)
                points.append((sample["t"], round(x_px), round(y_px)))
            if len(points) > 16:
                step = (len(points) - 1) / 15.0
                points = [points[int(i * step)] for i in range(16)]
            crop_keypoints = points

        # Caption placement: keep captions away from the primary face band.
        caption_band = "bottom"
        if center_samples:
            ys = sorted(sm["y"] for sm in center_samples)
            median_y = ys[len(ys) // 2]
            if median_y > 0.58:
                caption_band = "top"

        final_clip = dict(clip)
        final_clip.pop("zoom_intents", None)
        final_clip["duration"] = length
        final_clip["editing"] = {
            "aspect_ratio": geometry["target_label"],
            "zoom_enabled": bool(zoom_windows),
            "zoom_cuts": zoom_cuts,
            "total_magnification": round(s * max([z["zoom_level"] for z in zoom_windows] or [1.0]), 3),
            "manual_transform": manual_transform,
        }
        clips_final.append(final_clip)
        render_clips[str(index)] = {
            "zoom_windows": zoom_windows,
            "crop_keypoints": crop_keypoints,
            "caption_band": caption_band,
        }

        status = (f"clip {index}: {len(zoom_windows)} zoom window(s) kept"
                  if zoom_windows else f"clip {index}: NO zoom (correct default)")
        if zoom_windows:
            status += " [" + "; ".join(
                f"{z['start_rel_sec']:.1f}-{z['end_rel_sec']:.1f}s @{z['zoom_level']}x "
                f"({z['steering']})" for z in zoom_windows) + "]"
        if dropped_reasons:
            status += f"; dropped intents: {dropped_reasons}"
        logger.info(status)

    render_hints = {
        "geometry": {
            "mode": mode, "src_w": src_w, "src_h": src_h,
            "out_w": out_w, "out_h": out_h,
            "target_label": geometry["target_label"], "z_max": round(z_max, 3),
            "aspect_window": {"w": aw, "h": ah},
            "manual_transform": manual_transform,
            "manual_window": geometry["manual_window"],
            "output_quality": geometry["output_quality"],
            "quality_note": geometry["quality_note"],
        },
        "clips": render_clips,
    }

    total_zooms = sum(len(c["editing"]["zoom_cuts"]) for c in clips_final)
    logger.info("Zoom reconciliation complete: %d zoom window(s) across %d clip(s). Empty zoom "
                "means NO zoom - there is no fallback. Manual base framing: %s.",
                total_zooms, len(clips_final),
                f"active (scale {s:g})" if manual_active else "none")
    return clips_final, render_hints
