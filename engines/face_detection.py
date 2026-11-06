"""Face detection engine (port of clipzy_vision.py + PY_FACE_SCAN + PY_TRACK).

YuNet (OpenCV Zoo, 2023mar) is the primary detector; the Haar cascade is the
fallback. The coarse scan samples the vision engine's coarse grid; the clip
tracking stage samples at 2 fps inside each selected clip with the IoU-based
tracker, histogram scene-cut detection, primary-subject selection
(persistence x area x centrality, cut-spanning penalized) and the EMA-smoothed
center samples used by zoom steering and subject-aware crop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2

from cache import cache_manager
from config import VISION_CACHE_VER, YUNET_URL
from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("face_detection")


# ---------------------------------------------------------------------------
# Detector loading (YuNet preferred, Haar fallback)
# ---------------------------------------------------------------------------
def ensure_yunet_model(models_dir: Path) -> Path | None:
    """Restore from the resource cache or download + validate the YuNet model."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "yunet.onnx"

    def yunet_ready(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 100_000
        except OSError:
            return False

    if yunet_ready(target):
        logger.info("YuNet model restored from the resource cache: %d bytes.",
                    target.stat().st_size)
        return target

    import urllib.request
    try:
        with urllib.request.urlopen(YUNET_URL, timeout=90) as response:
            data = response.read()
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        if yunet_ready(tmp):
            tmp.replace(target)
            logger.info("YuNet model downloaded (%d bytes) and added to the resource cache.",
                        target.stat().st_size)
            return target
        logger.warning("YuNet model download failed validation; the Haar cascade will be used.")
        tmp.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("YuNet model unavailable (%s); the Haar cascade will be used.", exc)
        target.unlink(missing_ok=True)
    return None


class FaceEngine:
    """Loaded detector pair (method name + callable), mirroring clipzy_vision."""

    def __init__(self, models_dir: Path, score_threshold: float = 0.6) -> None:
        model_path = ensure_yunet_model(models_dir)
        self.method = "haar"
        self._detector = None
        if model_path is not None and model_path.stat().st_size > 100_000:
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(model_path), "", (320, 320), score_threshold, 0.3, 5000)
                self.method = "yunet"
            except cv2.error:
                self._detector = None
        if self._detector is None:
            self._detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        logger.info("Face detector ready: %s.", self.method)

    # ------------------------------------------------------------------
    def detect_faces(self, image):
        """Return [(x, y, w, h, confidence)] in image pixel coordinates."""
        height, width = image.shape[:2]
        boxes = []
        if self.method == "yunet":
            scale = 1.0
            working = image
            if max(width, height) > 640:
                scale = 640.0 / max(width, height)
                working = cv2.resize(image, (int(width * scale), int(height * scale)),
                                     interpolation=cv2.INTER_AREA)
            wh, ww = working.shape[:2]
            self._detector.setInputSize((ww, wh))
            retval, faces = self._detector.detect(working)
            if retval is not None and faces is not None:
                for face in faces:
                    x, y, w, h = (float(v) / scale for v in face[:4])
                    score = float(face[14]) if len(face) > 14 else 0.0
                    boxes.append((x, y, w, h, score))
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self._detector.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=5,
                minSize=(max(24, int(min(height, width) * 0.06)),) * 2,
            )
            if rects is not None:
                for (x, y, w, h) in rects:
                    boxes.append((float(x), float(y), float(w), float(h), 0.55))
        return boxes


class IoUTracker:
    """Minimal IoU-based face tracker with stable IDs and an aging policy."""

    def __init__(self, max_age_seconds=1.0):
        self.max_age = max_age_seconds
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, t, detections):
        """detections: [(x, y, w, h, conf)]. Returns [(id, box, conf)] after matching."""
        assigned = set()
        results = []
        for track in self.tracks:
            best, best_iou = None, 0.25
            for index, (x, y, w, h, conf) in enumerate(detections):
                if index in assigned:
                    continue
                iou = self._iou(track["box"], (x, y, w, h))
                if iou > best_iou:
                    best, best_iou = index, iou
            if best is not None:
                x, y, w, h, conf = detections[best]
                track["box"] = (x, y, w, h)
                track["last_t"] = t
                track["missed_since"] = None
                track["history"].append((t, x, y, w, h, conf))
                assigned.add(best)
                results.append((track["id"], (x, y, w, h), conf))
        for track in self.tracks:
            if track["missed_since"] is None and track["last_t"] != t:
                track["missed_since"] = t
        self.tracks = [tr for tr in self.tracks
                       if tr["missed_since"] is None or (t - tr["missed_since"]) <= self.max_age]
        for index, (x, y, w, h, conf) in enumerate(detections):
            if index in assigned:
                continue
            track = {"id": self.next_id, "box": (x, y, w, h), "last_t": t,
                     "missed_since": None, "history": [(t, x, y, w, h, conf)]}
            self.next_id += 1
            self.tracks.append(track)
            results.append((track["id"], (x, y, w, h), conf))
        return results


def histogram_correlation(prev_gray, gray):
    """Scene-cut helper: histogram correlation between consecutive frames."""
    if prev_gray is None:
        return 1.0
    hist_a = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


# ---------------------------------------------------------------------------
# Coarse face scan (PY_FACE_SCAN)
# ---------------------------------------------------------------------------
def coarse_face_scan(source: dict, frames_manifest: dict, engine: FaceEngine,
                     settings) -> dict:
    cache_paths = cache_manager.CachePaths(settings.cache_dir, source["source_sha256"])
    cached = cache_manager.load_cached(
        cache_paths.face_context(),
        expect={"source_sha256": source.get("source_sha256"),
                "pipeline": VISION_CACHE_VER, "detector": engine.method},
        ttl_days=30)
    if cached is not None:
        logger.info("CACHE HIT: coarse face scan reused for source sha256:%s (detector %s).",
                    source["source_sha256"][:12], engine.method)
        return cached

    interval = frames_manifest["coarse_interval"]
    samples = []
    faces_total = 0
    for index in range(frames_manifest["coarse_count"]):
        path = Path(cache_paths.coarse_dir()) / f"coarse_{index + 1:05d}.jpg"
        if not path.is_file():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = engine.detect_faces(image)
        faces = []
        for (x, y, w, h, conf) in boxes:
            faces.append({
                "x": round((x + w / 2) / width, 4),
                "y": round((y + h / 2) / height, 4),
                "w": round(w / width, 4),
                "h": round(h / height, 4),
                "conf": round(float(conf), 3),
            })
        faces_total += len(faces)
        samples.append({"t": round(index * interval, 2), "faces": faces})

    frames_with_faces = sum(1 for s in samples if s["faces"])
    pct = (100.0 * frames_with_faces / len(samples)) if samples else 0.0
    avg = (faces_total / len(samples)) if samples else 0.0

    context = {
        "method": engine.method,
        "sample_count": len(samples),
        "frames_with_faces_pct": round(pct, 1),
        "avg_faces_per_frame": round(avg, 2),
        "samples": samples,
    }
    cache_manager.save_cached(
        cache_paths.face_context(), context,
        meta={"source_sha256": source.get("source_sha256"),
              "pipeline": VISION_CACHE_VER, "detector": engine.method})
    logger.info("Coarse face scan (%s): faces visible in %.0f%% of %d samples, "
                "average %.1f face(s) per frame.", engine.method, pct, len(samples), avg)
    return context


# ---------------------------------------------------------------------------
# In-clip face tracking (PY_TRACK)
# ---------------------------------------------------------------------------
def track_faces_in_clips(curation: dict, engine: FaceEngine, workspace) -> list[dict]:
    ffmpeg = ff.ffmpeg_bin()
    sample_fps = 2.0
    track_root = workspace.work / "track"
    track_root.mkdir(parents=True, exist_ok=True)

    tracks_out = []
    for clip in curation["clips"]:
        index = clip["clip_index"]
        start = clip["start_sec"]
        length = round(clip["end_sec"] - clip["start_sec"], 3)
        frame_dir = track_root / f"c{index}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(workspace.source),
            "-an", "-vf", f"fps={sample_fps},scale=640:-2:flags=bicubic", "-q:v", "5",
            str(frame_dir / "f_%05d.jpg"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning("frame dump failed for clip %d: %s", index, proc.stderr[-300:])
            tracks_out.append({"clip_index": index, "method": engine.method,
                               "sample_fps": sample_fps, "samples": [], "cuts": [],
                               "primary": None})
            continue

        files = sorted(f.name for f in frame_dir.glob("*.jpg"))
        samples = []
        tracker = IoUTracker(max_age_seconds=1.0)
        prev_gray = None
        cuts = []
        history = {}

        for position, name in enumerate(files):
            image = cv2.imread(str(frame_dir / name))
            if image is None:
                continue
            t = position / sample_fps
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if histogram_correlation(prev_gray, gray) < 0.5:
                cuts.append(round(t, 2))
            prev_gray = gray
            height, width = image.shape[:2]
            boxes = engine.detect_faces(image)
            matched = tracker.update(t, boxes)
            faces = []
            for track_id, (x, y, w, h), conf in matched:
                faces.append({
                    "id": track_id,
                    "x": round((x + w / 2) / width, 4),
                    "y": round((y + h / 2) / height, 4),
                    "w": round(w / width, 4),
                    "h": round(h / height, 4),
                    "conf": round(float(conf), 3),
                })
                history.setdefault(track_id, []).append((t, x, y, w, h, conf))
            samples.append({"t": round(t, 2), "faces": faces})

        cut_set = sorted(set(cuts))
        frame_count = max(1, len(samples))

        # Primary subject = persistence x area x centrality; tracks spanning a
        # scene cut are penalized because the framing must never jump across a cut.
        best_id, best_score = None, -1.0
        ids = sorted({f["id"] for s in samples for f in s["faces"]})
        for track_id in ids:
            entries = [(s["t"], f) for s in samples for f in s["faces"] if f["id"] == track_id]
            persistence = len(entries) / frame_count
            if persistence < 0.4:
                continue
            median_area = sorted(f["w"] * f["h"] for _, f in entries)[len(entries) // 2]
            median_x = sorted(f["x"] for _, f in entries)[len(entries) // 2]
            median_y = sorted(f["y"] for _, f in entries)[len(entries) // 2]
            centrality = 1.0 - min(1.0, ((median_x - 0.5) ** 2 + (median_y - 0.5) ** 2) ** 0.5 * 2)
            score = 0.5 * persistence + 0.3 * min(1.0, median_area * 8) + 0.2 * centrality
            if any(entries[0][0] < cut < entries[-1][0] for cut in cut_set):
                score *= 0.4
            if score > best_score:
                best_id, best_score = track_id, score

        primary = None
        if best_id is not None:
            raw_centers = [(s["t"], f["x"], f["y"]) for s in samples
                           for f in s["faces"] if f["id"] == best_id]
            center_samples = []
            ema = None
            for i, (t, cx, cy) in enumerate(raw_centers):
                lo, hi = max(0, i - 2), min(len(raw_centers), i + 3)
                mx = sorted(p[1] for p in raw_centers[lo:hi])[(hi - lo) // 2]
                my = sorted(p[2] for p in raw_centers[lo:hi])[(hi - lo) // 2]
                if ema is None:
                    ema = [mx, my]
                else:
                    ema[0] = 0.4 * mx + 0.6 * ema[0]
                    ema[1] = 0.4 * my + 0.6 * ema[1]
                center_samples.append({"t": round(t, 2), "x": round(ema[0], 4), "y": round(ema[1], 4)})
            coverage = len(center_samples) / frame_count
            primary = {"id": best_id, "coverage": round(coverage, 3), "center_samples": center_samples}

        tracks_out.append({
            "clip_index": index,
            "method": engine.method,
            "sample_fps": sample_fps,
            "samples": samples,
            "cuts": cut_set,
            "primary": primary,
        })
        logger.info("Clip %d: %d tracked frames, %d face track(s), primary=%s, "
                    "coverage=%s, cuts=%d.", index, len(samples), len(ids),
                    f"#{best_id}" if best_id is not None else "none",
                    (primary or {}).get("coverage", 0), len(cut_set))

    detected = sum(1 for t in tracks_out if t["primary"])
    logger.info("Detailed face tracking complete: primary subject tracked in %d/%d clip(s) "
                "using %s.", detected, len(tracks_out), engine.method)
    return tracks_out
