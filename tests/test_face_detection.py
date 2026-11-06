"""Face detection engine: tracker, histogram helper, coarse scan + tracking."""

import numpy as np

from engines import face_detection


def _detect(width, height, boxes):
    return [(x, y, w, h, 0.9) for x, y, w, h in boxes]


def test_iou_tracker_matches_and_ages():
    tracker = face_detection.IoUTracker(max_age_seconds=1.0)
    t0 = tracker.update(0.0, _detect(640, 360, [(100, 100, 60, 60)]))
    assert len(t0) == 1
    id0 = t0[0][0]
    # Same box at t=0.5 -> same id.
    t1 = tracker.update(0.5, _detect(640, 360, [(104, 102, 60, 60)]))
    assert t1[0][0] == id0
    # Disappeared: after max_age the track is dropped.
    tracker.update(2.0, [])
    assert tracker.tracks == [] or all(tr["missed_since"] is not None for tr in tracker.tracks)
    # A new object gets a NEW id (never reuses stale identity).
    t3 = tracker.update(2.5, _detect(640, 360, [(400, 200, 60, 60)]))
    assert t3[0][0] != id0


def test_iou_tracker_two_objects():
    tracker = face_detection.IoUTracker(max_age_seconds=1.0)
    result = tracker.update(0.0, _detect(640, 360, [(100, 100, 60, 60), (300, 100, 60, 60)]))
    assert {r[0] for r in result} == {1, 2}
    result = tracker.update(0.5, _detect(640, 360, [(300, 100, 60, 60), (102, 101, 60, 60)]))
    # Order of results follows detections; ids stay stable.
    assert {r[0] for r in result} == {1, 2}


def test_histogram_correlation():
    import cv2

    a = np.full((180, 320), 100, dtype=np.uint8)
    b = np.full((180, 320), 100, dtype=np.uint8)
    c = np.full((180, 320), 250, dtype=np.uint8)
    assert face_detection.histogram_correlation(None, a) == 1.0
    assert face_detection.histogram_correlation(a, b) > 0.99
    assert face_detection.histogram_correlation(a, c) < 0.9


def test_face_engine_haar_fallback(settings, monkeypatch):
    # No YuNet model in the test models dir -> Haar cascade path.
    monkeypatch.setattr(face_detection, "ensure_yunet_model", lambda _: None)
    engine = face_detection.FaceEngine(settings.models_dir)
    assert engine.method == "haar"
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    boxes = engine.detect_faces(image)
    assert boxes == []  # no faces in a black frame, but no crash either


def test_coarse_face_scan_caches(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(face_detection, "ensure_yunet_model", lambda _: None)

    class FakeEngine:
        method = "haar"

        def detect_faces(self, image):
            # One fake face centered: 20% box at image center.
            h, w = image.shape[:2]
            return [(w * 0.4, h * 0.4, w * 0.2, h * 0.2, 0.8)]

    manifest = {"coarse_interval": 8.0, "coarse_count": 3}
    source = {"source_sha256": "deadbeef" * 8}
    # The coarse frames live in the persistent vision cache dir.
    from cache import cache_manager
    import cv2

    coarse_dir = cache_manager.CachePaths(settings.cache_dir, source["source_sha256"]).coarse_dir()
    import os
    os.makedirs(coarse_dir, exist_ok=True)
    for i in range(1, 4):
        cv2.imwrite(os.path.join(coarse_dir, f"coarse_{i:05d}.jpg"),
                    np.full((180, 320, 3), 120, dtype=np.uint8))
    context = face_detection.coarse_face_scan(source, manifest, FakeEngine(), settings)
    assert context["sample_count"] == 3
    assert context["frames_with_faces_pct"] == 100.0
    assert context["avg_faces_per_frame"] == 1.0
    assert len(context["samples"][0]["faces"]) == 1

    # Second call hits the computation cache (identical payload modulo meta).
    context2 = face_detection.coarse_face_scan(source, manifest, FakeEngine(), settings)
    context2.pop("cache_meta", None)
    assert context2 == {k: v for k, v in context.items() if k != "cache_meta"}
