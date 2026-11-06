"""The unified rendering transform chain + FFmpeg argv construction.

Port of the transform builders in the workflow's PY_PLAN:

* ramp_window   - cosine-ramped bump in [0,1] over [ws,we] with 0.8s ease;
* center_expr   - piecewise-linear interpolation with hold at both ends;
* build_zoom_chain - stage 2: dynamic punch-in INSIDE the manual window
  (lanczos), face-follow anchored; neutral=True builds the identical chain
  with zero zoom amplitude for QC comparison;
* build_manual_crop - stage 1: the manual pinch-zoom window as static crop;
* build_aspect_crop - stage 4: largest target-ratio rect inside the manual
  window, optionally steered by the tracked primary face;
* build_pad_graph - stage 4 pad mode: blurred-background canvas;
* build_render_argv - the full single-pass render command (H.264 CRF 18
  baseline; NVENC variant selected by utils.ffmpeg.choose_encoder when a
  GPU encoder is available - output stays H.264/yuv420p/AAC either way).
"""

from __future__ import annotations

import os

from utils import ffmpeg as ff




def filter_working_directory(*paths) -> str:
    """Return a shared cwd so FFmpeg filter paths can be relative.

    FFmpeg filtergraphs parse Windows drive letters, backslashes and spaces
    differently from normal argv paths. Relative paths avoid that ambiguity.
    """
    existing = [os.path.abspath(os.fspath(p)) for p in paths if p]
    if not existing:
        return os.getcwd()
    return os.path.commonpath(existing)


def _filter_relative_path(path, base_dir) -> str:
    """Format a path for an FFmpeg filtergraph using a relative POSIX path."""
    value = os.path.relpath(os.path.abspath(os.fspath(path)), base_dir)
    value = value.replace("\\\\", "/").replace("\\", "/")
    # Single-quote the option value; escape only characters special inside
    # FFmpeg's filter-option quoting layer.
    value = value.replace("'", "\\\\'")
    return "'" + value + "'"


def _subtitle_filter(ass_path, fonts_dir, base_dir) -> str:
    ass = _filter_relative_path(ass_path, base_dir)
    fonts = _filter_relative_path(fonts_dir, base_dir)
    return f"subtitles=filename={ass}:fontsdir={fonts}"


def fnum(value) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def ramp_window(ws, we, amp) -> str:
    """Cosine-ramped bump in [0,1] over [ws,we] with 0.8s ease in/out."""
    ramp = min(0.8, (we - ws) / 3.0)
    a, b = fnum(ws), fnum(we)
    r = fnum(ramp)
    expr = (
        f"if(lt(t,{a}),0,"
        f"if(lt(t,{fnum(ws + ramp)}),(1-cos(PI*(t-{a})/{r}))/2,"
        f"if(lt(t,{fnum(we - ramp)}),1,"
        f"if(lt(t,{b}),(1-cos(PI*({b}-t)/{r}))/2,0))))"
    )
    return f"({amp}*({expr}))"


def center_expr(keypoints, default) -> str:
    """Piecewise-linear interpolation over [(t, value)] with hold at both ends."""
    by_time = {}
    for t, value in keypoints:
        by_time[round(float(t), 2)] = float(value)
    points = sorted(by_time.items())
    if not points:
        return fnum(default)
    if len(points) > 6:
        step = (len(points) - 1) / 5.0
        points = [points[int(i * step)] for i in range(6)]
    expr = fnum(points[-1][1])
    for i in range(len(points) - 1, 0, -1):
        t0, v0 = points[i - 1]
        t1, v1 = points[i]
        if t1 - t0 < 0.05:
            continue
        seg = f"{fnum(v0)}+({fnum(v1 - v0)})*(t-{fnum(t0)})/{fnum(t1 - t0)}"
        expr = f"if(lt(t,{fnum(t1)}),{seg},{expr})"
    t0, v0 = points[0]
    expr = f"if(lt(t,{fnum(t0)}),{fnum(v0)},{expr})"
    return f"({expr})"


def build_zoom_chain(clip_hints, mw, mh, neutral=False):
    """Stage 2 of the unified transform: dynamic punch-in INSIDE the manual
    window (lanczos), face-follow anchored. Returns a filter list or None."""
    windows = clip_hints.get("zoom_windows") or []
    if not windows:
        return None
    z_terms = []
    cx_points, cy_points = [], []
    for window in windows:
        amp = 0.0 if neutral else round(window["zoom_level"] - 1.0, 4)
        z_terms.append(ramp_window(window["start_rel_sec"], window["end_rel_sec"], fnum(amp)))
        for keypoint in window.get("keypoints") or []:
            t, x, y = keypoint
            cx_points.append((t, x))
            cy_points.append((t, y))
    if not z_terms:
        return None
    if neutral:
        return []
    z_expr = "1+" + "+".join(z_terms)
    cx = center_expr(cx_points, 0.5)
    cy = center_expr(cy_points, 0.5)
    scale = (f"scale=w='2*round(iw*({z_expr})/2)':h='2*round(ih*({z_expr})/2)':"
             f"eval=frame:flags=lanczos")
    # IMPORTANT: inside crop's x/y expressions, in_w/in_h reflect the
    # link-negotiated (first-frame) size, NOT the per-frame size coming out of
    # the dynamic scale. The zoomed dimensions are therefore computed
    # symbolically as mw*Z(t); clamping keeps the 1-px even-rounding difference
    # during ramps harmless.
    zoomed_w = f"{mw}*({z_expr})"
    zoomed_h = f"{mh}*({z_expr})"
    x_expr = (f"2*floor(min(max({cx}*({zoomed_w})-{mw}/2,0),"
              f"max(({zoomed_w})-{mw},0))/2)")
    y_expr = (f"2*floor(min(max({cy}*({zoomed_h})-{mh}/2,0),"
              f"max(({zoomed_h})-{mh},0))/2)")
    crop = f"crop=w={mw}:h={mh}:x='{x_expr}':y='{y_expr}'"
    return [scale, crop]


def build_manual_crop(manual_scale, mw, mh, mx, my, src_w, src_h):
    """Stage 1: the manual pinch-zoom window as a static base crop."""
    if manual_scale <= 1.001 and mx == 0 and my == 0 and (mw, mh) == (src_w, src_h):
        return None
    return f"crop=w={mw}:h={mh}:x={mx}:y={my}"


def build_aspect_crop(clip_hints, mode, mw, mh, aw, ah):
    """Stage 4: largest target-ratio rect inside the manual window, optionally
    steered by the tracked primary face (never overriding the manual framing)."""
    if mode != "crop":
        return None
    if (aw, ah) == (mw, mh):
        return None
    keypoints = clip_hints.get("crop_keypoints")
    if keypoints:
        x_expr = center_expr([(t, x) for t, x, y in keypoints], (mw - aw) / 2)
        y_expr = center_expr([(t, y) for t, x, y in keypoints], (mh - ah) / 2)
        x_full = f"2*floor(min(max({x_expr},0),{mw - aw})/2)"
        y_full = f"2*floor(min(max({y_expr},0),{mh - ah})/2)"
        return f"crop=w={aw}:h={ah}:x='{x_full}':y='{y_full}'"
    x0 = 2 * (int((mw - aw) / 2) // 2)
    y0 = 2 * (int((mh - ah) / 2) // 2)
    return f"crop=w={aw}:h={ah}:x={x0}:y={y0}"


def build_pad_graph(head, out_w, out_h):
    prefix = f"[0:v]{','.join(head)}," if head else "[0:v]"
    blur_radius = max(2, min(20, min(out_w, out_h) // 4))
    return (
        f"{prefix}split=2[fgs][bgs];"
        f"[bgs]scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={out_w}:{out_h},boxblur={blur_radius}:2,setsar=1[bgo];"
        f"[fgs]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"setsar=1[fgo];"
        f"[bgo][fgo]overlay=x=(W-w)/2:y=(H-h)/2[vout]"
    )


WATERMARK_SCALE_FACTORS = {"small": 0.12, "medium": 0.18, "large": 0.26}
WATERMARK_XY = {
    "top_left": "24:24",
    "top_right": "main_w-overlay_w-24:24",
    "bottom_left": "24:main_h-overlay_h-24",
    "bottom_right": "main_w-overlay_w-24:main_h-overlay_h-24",
}


def build_render_argv(clip, clip_hints, geometry, subtitles, params, source,
                      workspace, fonts_dir, watermark_file, encoder: dict) -> list[str]:
    """Assemble the full ffmpeg argv for one clip (single-pass unified chain)."""
    index = clip["clip_index"]
    start = clip["start_sec"]
    length = round(clip["end_sec"] - clip["start_sec"], 3)
    src_w, src_h = geometry["src_w"], geometry["src_h"]
    out_w, out_h = geometry["out_w"], geometry["out_h"]
    mode = geometry["mode"]
    aw, ah = geometry["aspect_window"]["w"], geometry["aspect_window"]["h"]
    manual_transform = geometry.get("manual_transform")
    manual_window = geometry.get("manual_window") or {"w": src_w, "h": src_h, "x": 0, "y": 0}
    mw, mh = manual_window["w"], manual_window["h"]
    mx, my = manual_window["x"], manual_window["y"]
    fps = source["fps"]
    has_audio = source["has_audio"]
    manual_scale = float((manual_transform or {}).get("scale", 1.0))

    encode_preset = "medium" if max(out_w, out_h) <= 1920 else "faster"
    gop = max(12, int(round(fps * 2)))

    sub = next((s for s in subtitles.get("clips", []) if s["clip_index"] == index), None)
    burned_in = bool(sub and sub.get("burned_in"))

    # ---- Unified transform chain (stages 1 -> 5) ------------------------
    manual_crop = build_manual_crop(manual_scale, mw, mh, mx, my, src_w, src_h)
    zoom_chain = build_zoom_chain(clip_hints, mw, mh)
    tail = []
    ass_path = (sub.get("file") if burned_in
                else str(workspace.work / "subs" / f"clip_{index}.ass"))
    filter_paths = [ass_path, fonts_dir]
    if watermark_file:
        filter_paths.append(watermark_file)
    filter_cwd = filter_working_directory(*filter_paths)
    if burned_in:
        tail.append(_subtitle_filter(ass_path, fonts_dir, filter_cwd))
    tail.append("setsar=1")
    tail.append("format=yuv420p")

    pre = ([manual_crop] if manual_crop else []) + (zoom_chain or [])
    if mode == "pad":
        graph = build_pad_graph(pre, out_w, out_h)
        graph += ";" + "[vout]" + ",".join(tail) + "[vfinal]"
    else:
        chain_parts = list(pre)
        aspect_crop = build_aspect_crop(clip_hints, mode, mw, mh, aw, ah)
        if aspect_crop:
            chain_parts.append(aspect_crop)
        current_w, current_h = (aw, ah) if aspect_crop else (mw, mh)
        if (current_w, current_h) != (out_w, out_h):
            chain_parts.append(f"scale={out_w}:{out_h}:flags=lanczos")
        chain_parts.extend(tail)
        graph = "[0:v]" + ",".join(chain_parts) + "[vfinal]"

    wm_file = str(watermark_file) if watermark_file else "work/watermark_logo.png"
    has_wm = bool(watermark_file) and os.path.exists(wm_file) and os.path.getsize(wm_file) > 0
    wm_pos = params.get("watermark_position", "bottom_right").lower()
    wm_size_key = params.get("watermark_size", "medium").lower()
    wm_scale_factor = WATERMARK_SCALE_FACTORS.get(wm_size_key, 0.18)
    wm_xy = WATERMARK_XY.get(wm_pos, WATERMARK_XY["bottom_right"])

    if has_wm:
        graph = graph.replace("[vfinal]", "[vpre]")
        wm_filter_path = _filter_relative_path(wm_file, filter_cwd)
        graph += f";movie={wm_filter_path},scale={out_w}*{wm_scale_factor}:-1[wm];[vpre][wm]overlay={wm_xy}[vfinal]"

    argv = [
        ff.ffmpeg_bin(), "-y", "-nostdin",
        "-hide_banner", "-loglevel", "warning",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(workspace.source),
        "-filter_complex", graph,
        "-map", "[vfinal]",
    ]
    if has_audio:
        argv += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]

    if encoder.get("gpu"):
        nvenc_preset = "p5" if encode_preset == "medium" else "p4"
        argv += [
            "-c:v", encoder["codec"],
            encoder["preset_flag"], nvenc_preset,
            *encoder["quality_flags"],
        ]
    else:
        argv += [
            "-c:v", encoder["codec"],
            encoder["preset_flag"], encode_preset,
            *encoder["quality_flags"],
        ]
    argv += [
        "-g", str(gop), "-movflags", "+faststart",
        str(workspace.output_clips / f"clip_{index}.mp4"),
    ]
    return argv


def build_graph_variants(clip, clip_hints, geometry, subtitles, params, workspace,
                         fonts_dir, watermark_file) -> dict:
    """Build the graph strings (video_graph, qc_graph, qc_nozoom_graph) used by
    the quality-control reference renders."""
    index = clip["clip_index"]
    src_w, src_h = geometry["src_w"], geometry["src_h"]
    out_w, out_h = geometry["out_w"], geometry["out_h"]
    mode = geometry["mode"]
    aw, ah = geometry["aspect_window"]["w"], geometry["aspect_window"]["h"]
    manual_transform = geometry.get("manual_transform")
    manual_window = geometry.get("manual_window") or {"w": src_w, "h": src_h, "x": 0, "y": 0}
    mw, mh = manual_window["w"], manual_window["h"]
    mx, my = manual_window["x"], manual_window["y"]
    manual_scale = float((manual_transform or {}).get("scale", 1.0))

    sub = next((s for s in subtitles.get("clips", []) if s["clip_index"] == index), None)
    burned_in = bool(sub and sub.get("burned_in"))
    ass_path = (sub.get("file") if burned_in
                else str(workspace.work / "subs" / f"clip_{index}.ass"))

    manual_crop = build_manual_crop(manual_scale, mw, mh, mx, my, src_w, src_h)
    zoom_chain = build_zoom_chain(clip_hints, mw, mh)
    zoom_chain_neutral = build_zoom_chain(clip_hints, mw, mh, neutral=True)
    filter_paths = [ass_path, fonts_dir]
    if watermark_file:
        filter_paths.append(watermark_file)
    filter_cwd = filter_working_directory(*filter_paths)
    tail = []
    if burned_in:
        tail.append(_subtitle_filter(ass_path, fonts_dir, filter_cwd))
    tail.append("setsar=1")
    tail.append("format=yuv420p")

    pre = ([manual_crop] if manual_crop else []) + (zoom_chain or [])
    if mode == "pad":
        graph = build_pad_graph(pre, out_w, out_h)
        graph += ";" + "[vout]" + ",".join(tail) + "[vfinal]"
    else:
        chain_parts = list(pre)
        aspect_crop = build_aspect_crop(clip_hints, mode, mw, mh, aw, ah)
        if aspect_crop:
            chain_parts.append(aspect_crop)
        current_w, current_h = (aw, ah) if aspect_crop else (mw, mh)
        if (current_w, current_h) != (out_w, out_h):
            chain_parts.append(f"scale={out_w}:{out_h}:flags=lanczos")
        chain_parts.extend(tail)
        graph = "[0:v]" + ",".join(chain_parts) + "[vfinal]"

    wm_file = str(watermark_file) if watermark_file else ""
    has_wm = bool(watermark_file) and os.path.exists(wm_file) and os.path.getsize(wm_file) > 0
    if has_wm:
        wm_pos = params.get("watermark_position", "bottom_right").lower()
        wm_size_key = params.get("watermark_size", "medium").lower()
        wm_scale_factor = WATERMARK_SCALE_FACTORS.get(wm_size_key, 0.18)
        wm_xy = WATERMARK_XY.get(wm_pos, WATERMARK_XY["bottom_right"])
        graph = graph.replace("[vfinal]", "[vpre]")
        wm_filter_path = _filter_relative_path(wm_file, filter_cwd)
        graph += f";movie={wm_filter_path},scale={out_w}*{wm_scale_factor}:-1[wm];[vpre][wm]overlay={wm_xy}[vfinal]"

    # QC reference graphs: the identical video chain WITHOUT the subtitle
    # filter (qc_graph), plus a zoom-neutral variant (qc_nozoom_graph) whose
    # zoom amplitudes are zeroed - comparing the exported frame against both
    # proves the zoom really moved pixels instead of merely existing in
    # metadata.
    qc_graph = graph
    if burned_in:
        sub_filter = _subtitle_filter(ass_path, fonts_dir, filter_cwd)
        qc_graph = graph.replace(sub_filter + ",", "")
    neutral_pre = ([manual_crop] if manual_crop else []) + (zoom_chain_neutral or [])
    if mode == "pad":
        qc_nozoom_graph = build_pad_graph(neutral_pre, out_w, out_h)
        qc_nozoom_graph += ";" + "[vout]" + ",".join(
            t for t in tail if not t.startswith("subtitles=")) + "[vfinal]"
    else:
        neutral_parts = list(neutral_pre)
        aspect_crop = build_aspect_crop(clip_hints, mode, mw, mh, aw, ah)
        if aspect_crop:
            neutral_parts.append(aspect_crop)
        current_w, current_h = (aw, ah) if aspect_crop else (mw, mh)
        if (current_w, current_h) != (out_w, out_h):
            neutral_parts.append(f"scale={out_w}:{out_h}:flags=lanczos")
        neutral_parts.extend(t for t in tail if not t.startswith("subtitles="))
        qc_nozoom_graph = "[0:v]" + ",".join(neutral_parts) + "[vfinal]"

    return {"video_graph": graph, "qc_graph": qc_graph, "qc_nozoom_graph": qc_nozoom_graph}
