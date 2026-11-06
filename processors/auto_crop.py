"""Output geometry policy (the geometry half of the workflow's PY_ZOOM).

UNIFIED RENDERING TRANSFORM - stage map (single FFmpeg pass contract):
  1. MANUAL BASE FRAMING  - static crop of the source to the user's
     pinch-zoom window (scale s, offsets). Defines the "camera".
  2. AI AUTO ZOOM          - time-varying punch-in INSIDE the manual window
     (2-15s, cosine ramps, face-follow anchor when a primary face track
     exists, centered otherwise).
  3. FACE TRACKING         - only steers the anchor of stage 2 and the
     aspect-crop center of stage 4; it never overrides stage 1.
  4. ASPECT-RATIO CROP/PAD - largest target-ratio rect inside the manual
     window (crop mode) or blurred-background pad.
  5. FINAL SCALE           - lanczos resize to the output canvas.
     Subtitles are burned AFTER all geometry so they never zoom.
The final crop always stays inside source boundaries; manual and automatic
transforms compose multiplicatively (s * z(t)).
"""

from __future__ import annotations

from utils.logging_setup import get_logger

logger = get_logger("geometry")


def parse_ratio(label: str) -> float:
    if ":" in label:
        a, b = label.split(":", 1)
        return float(a) / float(b)
    return float(label)


def round_even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def compute_geometry(source: dict, params: dict, face_context: dict) -> dict:
    """Compute the full output geometry + aspect conversion mode."""
    src_w, src_h = source["width"], source["height"]
    src_ratio = src_w / src_h
    aspect = params["aspect_ratio"]
    preserve_quality = params["preserve_source_quality"] == "true"
    output_quality = params.get("output_quality", "original")
    manual_transform = params.get("manual_transform")

    # ---- Stage 1: the manual pinch-zoom window ---------------------------
    manual_active = bool(manual_transform)
    if manual_active:
        s = float(manual_transform["scale"])
        # Window center in source-fraction coords, clamped so the window never
        # leaves the frame (the Android preview clamps identically, so preview
        # and export agree by construction).
        cx_w = 0.5 + float(manual_transform["offset_x"])
        cy_w = 0.5 + float(manual_transform["offset_y"])
        mw_f, mh_f = 1.0 / s, 1.0 / s
        cx_w = min(1.0 - mw_f / 2, max(mw_f / 2, cx_w))
        cy_w = min(1.0 - mh_f / 2, max(mh_f / 2, cy_w))
        mw = round_even(src_w / s)
        mh = round_even(src_h / s)
        mx = round_even(min(max(cx_w * src_w - mw / 2, 0), src_w - mw))
        my = round_even(min(max(cy_w * src_h - mh / 2, 0), src_h - mh))
        logger.info("Manual base framing: window %dx%d at (%d,%d) (scale %g, offsets (%s, %s)).",
                    mw, mh, mx, my, s, manual_transform["offset_x"], manual_transform["offset_y"])
    else:
        s = 1.0
        mw, mh, mx, my = src_w, src_h, 0, 0

    if aspect == "source":
        target_ratio = src_ratio
        target_label = source["aspect_ratio"]
    else:
        target_ratio = parse_ratio(aspect)
        target_label = aspect

    # ---- Output canvas policy (output_quality) ---------------------------
    # original : native-ratio canvas that never upscales the source.
    # 1080p/720p/480p : short-side budget, capped at what the source can
    # support (explicit notice, never a silent replacement); upscaling is only
    # allowed when the user opted out of preserve_source_quality.
    native_w = round_even(min(src_w, src_h * target_ratio))
    native_h = round_even(native_w / target_ratio)
    if native_h > src_h:
        native_h = round_even(src_h)
        native_w = round_even(native_h * target_ratio)
    native_short = min(native_w, native_h)

    if output_quality == "original":
        out_w, out_h = native_w, native_h
        quality_note = "original (native dimensions, never upscaled)"
    else:
        requested_short = int(output_quality.replace("p", ""))
        if requested_short <= native_short:
            quality_note = f"{output_quality} as requested"
        elif preserve_quality:
            quality_note = (f"requested {output_quality} exceeds the source capability "
                            f"({native_short}px short side); rendering at native "
                            f"{native_short}px instead of upscaling")
        else:
            quality_note = (f"requested {output_quality} exceeds the source capability "
                            f"({native_short}px short side); upscaling is allowed because "
                            f"preserve_source_quality=false")
        budget_short = (requested_short if (requested_short <= native_short or not preserve_quality)
                        else native_short)
        if target_ratio >= 1.0:
            out_w = round_even(budget_short * target_ratio)
            out_h = round_even(budget_short)
        else:
            out_w = round_even(budget_short)
            out_h = round_even(budget_short / target_ratio)

    # ---- Stage 4 decision: crop vs pad ------------------------------------
    mode = "passthrough"
    if manual_active or abs(target_ratio - src_ratio) >= 0.01 or (out_w, out_h) != (src_w, src_h):
        # Largest target-ratio rect inside the MANUAL WINDOW; the final scale
        # stage maps it onto the canvas.
        aw = round_even(min(mw, mh * target_ratio))
        ah = round_even(aw / target_ratio)
        if ah > mh:
            ah = round_even(mh)
            aw = round_even(ah * target_ratio)
        w_frac, h_frac = aw / mw, ah / mh
        centers = []
        for sample in face_context.get("samples", []):
            for f in sample.get("faces", []):
                # Remap source-fraction coords into the manual window.
                x_m = (f["x"] * src_w - mx) / mw
                y_m = (f["y"] * src_h - my) / mh
                centers.append((x_m, y_m))
        if centers:
            inside = sum(1 for x_m, y_m in centers
                         if 0.5 - w_frac / 2 <= x_m <= 0.5 + w_frac / 2
                         and 0.5 - h_frac / 2 <= y_m <= 0.5 + h_frac / 2)
            containment = inside / len(centers)
            # Verified subjects: crop to them regardless of how narrow the
            # aspect window is (this is the standard shorts conversion).
            mode = "crop" if containment >= 0.75 else "pad"
        else:
            # No face evidence at all: only moderate blind crops fill the
            # frame; extreme conversions keep the full content on a blurred
            # canvas instead of guessing.
            containment = None
            mode = "crop" if (w_frac >= 0.5 and h_frac >= 0.5) else "pad"
        containment_note = "no face evidence" if containment is None else f"{containment:.0%}"
        logger.info("Aspect conversion %dx%d (window %dx%d) -> %dx%d: %s mode (face "
                    "containment %s, aspect window keeps %.0f%% width / %.0f%% height of the "
                    "manual window).", src_w, src_h, mw, mh, out_w, out_h, mode,
                    containment_note, w_frac * 100, h_frac * 100)
    else:
        aw, ah = src_w, src_h
        logger.info("Aspect: source ratio %s preserved; no geometric transform, no resizing "
                    "(output %dx%d).", target_label, out_w, out_h)

    # ---- AI zoom budget ----------------------------------------------------
    # The effective source for zooming is the manual window: total magnification
    # is manual_scale * ai_zoom_level(t). With a manual zoom the user already
    # accepted upscaling, so at least the standard 1.10 emphasis level stays
    # available (logged, never silent).
    min_m = min(mw, mh)
    min_out = min(out_w, out_h)
    upscale_budget = 1.25 if min_m >= 720 else 1.10
    z_max = min(1.20, upscale_budget * min_m / min_out)
    if manual_active:
        z_max = max(1.10, z_max)
    zoom_globally_enabled = z_max >= 1.06
    if not zoom_globally_enabled:
        logger.warning("zoom disabled globally: effective resolution budget %.2fx is below "
                       "1.06x (window %dx%d vs canvas %dx%d).", z_max, mw, mh, out_w, out_h)
    logger.info("Output quality: %s; AI zoom level cap %.2fx (total magnification ceiling "
                "%.2fx incl. manual scale %g).", quality_note, z_max, s * z_max, s)

    return {
        "mode": mode, "src_w": src_w, "src_h": src_h,
        "out_w": out_w, "out_h": out_h,
        "target_label": target_label, "z_max": round(z_max, 3),
        "zoom_globally_enabled": zoom_globally_enabled,
        "aspect_window": {"w": aw, "h": ah},
        "manual_transform": manual_transform,
        "manual_scale": s,
        "manual_window": {"w": mw, "h": mh, "x": mx, "y": my} if manual_active else None,
        "output_quality": output_quality,
        "quality_note": quality_note,
    }
