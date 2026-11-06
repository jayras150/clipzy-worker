"""Geometry policy + zoom reconciliation tests."""

from processors import auto_crop, auto_zoom


def _source(w=640, h=360, duration=120.0, fps=30.0):
    return {"width": w, "height": h, "duration": duration, "fps": fps,
            "aspect_ratio": "16:9", "vcodec": "h264", "pix_fmt": "yuv420p",
            "has_audio": True, "source_sha256": "x" * 64}


def _params(**overrides):
    params = {"aspect_ratio": "source", "preserve_source_quality": "true",
              "output_quality": "original", "manual_transform": None}
    params.update(overrides)
    return params


def _no_face_context():
    return {"samples": []}


def test_passthrough_geometry(sample_geometry_source=None):
    geometry = auto_crop.compute_geometry(_source(), _params(), _no_face_context())
    assert geometry["mode"] == "passthrough"
    assert (geometry["out_w"], geometry["out_h"]) == (640, 360)
    assert geometry["z_max"] >= 1.06
    assert geometry["manual_window"] is None


def test_quality_policy_never_upscales_by_default():
    geometry = auto_crop.compute_geometry(_source(), _params(output_quality="1080p"),
                                          _no_face_context())
    # 360p source cannot reach 1080p: capped at native, explicit note.
    assert min(geometry["out_w"], geometry["out_h"]) == 360
    assert "exceeds the source capability" in geometry["quality_note"]


def test_quality_policy_upscales_when_allowed():
    geometry = auto_crop.compute_geometry(
        _source(), _params(output_quality="1080p", preserve_source_quality="false"),
        _no_face_context())
    assert min(geometry["out_w"], geometry["out_h"]) == 1080


def test_quality_policy_downscales_to_request():
    geometry = auto_crop.compute_geometry(_source(1920, 1080), _params(output_quality="720p"),
                                          _no_face_context())
    assert (geometry["out_w"], geometry["out_h"]) == (1280, 720)


def test_extreme_aspect_without_faces_pads():
    # 16:9 -> 9:16 with no face evidence: mode pad (no blind guessing).
    geometry = auto_crop.compute_geometry(_source(), _params(aspect_ratio="9:16"),
                                          _no_face_context())
    assert geometry["mode"] == "pad"
    assert geometry["out_w"] < geometry["out_h"]


def test_moderate_aspect_without_faces_crops():
    geometry = auto_crop.compute_geometry(_source(), _params(aspect_ratio="4:3"),
                                          _no_face_context())
    # 4:3 window keeps 75% width / 100% height -> >= 0.5 both -> crop.
    assert geometry["mode"] == "crop"


def test_faces_drive_crop_mode():
    face_context = {"samples": [
        {"t": 0, "faces": [{"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "conf": 0.9}]},
    ] * 10}
    geometry = auto_crop.compute_geometry(_source(), _params(aspect_ratio="9:16"),
                                          face_context)
    # Centered subject verified by CV: crop to the subject (shorts conversion).
    assert geometry["mode"] == "crop"


def test_manual_transform_window():
    params = _params(manual_transform={"scale": 2.0, "offset_x": 0.1, "offset_y": 0.0})
    geometry = auto_crop.compute_geometry(_source(), params, _no_face_context())
    window = geometry["manual_window"]
    assert window["w"] == 320 and window["h"] == 180
    assert window["x"] > 0  # shifted right by the positive offset
    assert geometry["manual_scale"] == 2.0
    # Manual zoom guarantees at least the standard 1.10 AI zoom level.
    assert geometry["z_max"] >= 1.10


def test_zoom_reconciliation_drops_and_merges():
    source = _source(duration=60.0)
    geometry = auto_crop.compute_geometry(source, _params(), _no_face_context())
    curation = {"clips": [{
        "clip_index": 1, "start_sec": 0.0, "end_sec": 40.0, "viral_score": 80,
        "topic": "t", "hook": "", "title": "T", "description": "D", "tags": ["#a", "#b", "#c"],
        "visual_analysis": {"main_subject": "s", "scene": "c"},
        "zoom_intents": [
            {"start_rel_sec": 1.0, "end_rel_sec": 4.0, "zoom_level": 1.15,
             "reason": "emphasis"},
            # Overlaps within 1.6s of the previous window -> merged.
            {"start_rel_sec": 4.5, "end_rel_sec": 8.0, "zoom_level": 1.10,
             "reason": "reaction"},
            # Too short -> dropped.
            {"start_rel_sec": 20.0, "end_rel_sec": 21.0, "zoom_level": 1.15, "reason": "x"},
        ],
    }]}
    tracks = {1: {"clip_index": 1, "primary": None, "cuts": [], "sample_fps": 2.0,
                  "samples": []}}
    clips_final, hints = auto_zoom.reconcile_zooms(
        source, curation, tracks, _no_face_context(), _params(), geometry)

    assert len(clips_final) == 1
    editing = clips_final[0]["editing"]
    assert editing["zoom_enabled"] is True
    windows = hints["clips"]["1"]["zoom_windows"]
    assert len(windows) == 1  # merged into a single window
    assert windows[0]["start_rel_sec"] == 1.0
    assert windows[0]["end_rel_sec"] == 8.0
    assert windows[0]["zoom_level"] == 1.10  # 1.15 intent capped by the 360p resolution budget
    assert windows[0]["steering"].startswith("center (no tracked")
    assert editing["zoom_cuts"][0]["target"] == "primary_subject"
    assert editing["total_magnification"] >= 1.10

def test_zoom_reconciliation_no_intents_means_no_zoom():
    source = _source(duration=60.0)
    geometry = auto_crop.compute_geometry(source, _params(), _no_face_context())
    curation = {"clips": [{
        "clip_index": 1, "start_sec": 0.0, "end_sec": 30.0, "viral_score": 80,
        "topic": "t", "hook": "", "title": "T", "description": "D", "tags": ["#a", "#b", "#c"],
        "visual_analysis": {}, "zoom_intents": [],
    }]}
    tracks = {1: {"clip_index": 1, "primary": None, "cuts": [], "sample_fps": 2.0,
                  "samples": []}}
    clips_final, hints = auto_zoom.reconcile_zooms(
        source, curation, tracks, _no_face_context(), _params(), geometry)
    assert clips_final[0]["editing"]["zoom_enabled"] is False
    assert clips_final[0]["editing"]["zoom_cuts"] == []


def test_zoom_window_split_by_scene_cut():
    source = _source(duration=60.0)
    geometry = auto_crop.compute_geometry(source, _params(), _no_face_context())
    curation = {"clips": [{
        "clip_index": 1, "start_sec": 0.0, "end_sec": 40.0, "viral_score": 80,
        "topic": "t", "hook": "", "title": "T", "description": "D", "tags": ["#a", "#b", "#c"],
        "visual_analysis": {},
        "zoom_intents": [
            # A scene cut at t=12 inside a 5-20s window: keep the longest span.
            {"start_rel_sec": 5.0, "end_rel_sec": 20.0, "zoom_level": 1.10, "reason": "x"},
        ],
    }]}
    tracks = {1: {"clip_index": 1, "primary": None, "cuts": [12.0], "sample_fps": 2.0,
                  "samples": []}}
    _clips, hints = auto_zoom.reconcile_zooms(
        source, curation, tracks, _no_face_context(), _params(), geometry)
    window = hints["clips"]["1"]["zoom_windows"][0]
    assert window["start_rel_sec"] == 12.0 and window["end_rel_sec"] == 20.0


def test_caption_band_top_when_face_is_low():
    source = _source(duration=60.0)
    geometry = auto_crop.compute_geometry(source, _params(), _no_face_context())
    curation = {"clips": [{
        "clip_index": 1, "start_sec": 0.0, "end_sec": 30.0, "viral_score": 80,
        "topic": "t", "hook": "", "title": "T", "description": "D", "tags": ["#a", "#b", "#c"],
        "visual_analysis": {}, "zoom_intents": [],
    }]}
    center_samples = [{"t": i * 0.5, "x": 0.5, "y": 0.75} for i in range(20)]
    tracks = {1: {"clip_index": 1,
                  "primary": {"id": 1, "coverage": 1.0, "center_samples": center_samples},
                  "cuts": [], "sample_fps": 2.0, "samples": []}}
    _clips, hints = auto_zoom.reconcile_zooms(
        source, curation, tracks, _no_face_context(), _params(), geometry)
    assert hints["clips"]["1"]["caption_band"] == "top"
    # Face-follow steering is recorded for the (passthrough) render hints.
    assert hints["clips"]["1"]["zoom_windows"] == []
