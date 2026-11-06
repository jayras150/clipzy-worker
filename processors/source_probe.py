"""Source probing and fingerprinting.

Port of the workflow's `PY_PROBE`: FFprobe stream/format analysis with rotation
handling, FPS snapping, the 90-minute limit, aspect labeling, audio info and
the SHA256 content fingerprint that keys every computation cache.
"""

from __future__ import annotations

import hashlib
import math

from config import LONG_SOURCE_WARN_SECONDS, MAX_SOURCE_SECONDS
from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("source_probe")


class ProbeError(RuntimeError):
    pass


def sha256_file(path, block_size: int = 4 * 1024 * 1024) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest()


def probe_source(source_path) -> dict:
    data = ff.probe_media(source_path)
    if not data:
        raise ProbeError(f"ffprobe failed to inspect {source_path}.")

    video = None
    audio = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video is None:
            video = stream
        elif stream.get("codec_type") == "audio" and audio is None:
            audio = stream

    if video is None:
        raise ProbeError("the downloaded file contains no video stream.")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ProbeError("could not read video dimensions.")

    # Honor rotation metadata so the probe reflects display dimensions.
    rotation = 0
    for side_data in video.get("side_data_list") or []:
        if "rotation" in side_data:
            try:
                rotation += int(side_data["rotation"])
            except (TypeError, ValueError):
                pass
    rotation = abs(rotation) % 360
    if rotation in (90, 270):
        width, height = height, width
        logger.info("Rotation metadata detected (%d deg); display dimensions are %dx%d.",
                    rotation, width, height)

    def parse_rate(value):
        if not value:
            return None
        if "/" in value:
            num, den = value.split("/", 1)
            try:
                num, den = float(num), float(den)
                return num / den if den else None
            except ValueError:
                return None
        try:
            return float(value)
        except ValueError:
            return None

    fps = parse_rate(video.get("r_frame_rate")) or parse_rate(video.get("avg_frame_rate")) or 25.0
    snap_fps = [23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0, 90.0, 120.0]
    fps_snapped = fps
    for candidate in snap_fps:
        if abs(fps - candidate) <= 0.02:
            fps_snapped = candidate
            break

    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        try:
            duration = float(video.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    if duration <= 0:
        raise ProbeError("unable to read video duration.")

    if duration > MAX_SOURCE_SECONDS:
        raise ProbeError(
            f"source video is {duration / 60.0:.1f} minutes long; the pipeline limit is "
            f"{MAX_SOURCE_SECONDS // 60} minutes. Split the source or provide a shorter upload.")
    if duration > LONG_SOURCE_WARN_SECONDS:
        logger.warning("source is %.1f minutes long; this job may take a while.", duration / 60.0)

    gcd = math.gcd(width, height)
    ratio_w, ratio_h = width // gcd, height // gcd
    aspect_label = f"{width / height:.4f}" if ratio_h > 50 else f"{ratio_w}:{ratio_h}"

    video_bitrate = 0
    try:
        video_bitrate = int(video.get("bit_rate") or 0)
    except (TypeError, ValueError):
        video_bitrate = 0
    if not video_bitrate:
        try:
            video_bitrate = int(data.get("format", {}).get("bit_rate") or 0)
        except (TypeError, ValueError):
            video_bitrate = 0

    audio_info = None
    if audio is not None:
        audio_info = {
            "codec": audio.get("codec_name", "unknown"),
            "channels": int(audio.get("channels") or 0),
            "sample_rate": int(audio.get("sample_rate") or 0),
            "bitrate": int(audio.get("bit_rate") or 0),
        }

    source_sha256 = sha256_file(source_path)
    source = {
        "file": str(source_path),
        "width": width,
        "height": height,
        "fps": fps_snapped,
        "fps_exact": round(fps, 6),
        "rotation": rotation,
        "vcodec": video.get("codec_name", "unknown"),
        "pix_fmt": video.get("pix_fmt", "unknown"),
        "duration": round(duration, 3),
        "aspect_ratio": aspect_label,
        "aspect_value": round(width / height, 6),
        "has_audio": audio is not None,
        "audio": audio_info,
        "source_sha256": source_sha256,
        "video_bitrate": video_bitrate,
    }
    logger.info("Source: %dx%d @ %g fps, %s/%s, %.1fs, aspect %s, audio=%s",
                width, height, fps_snapped, source["vcodec"], source["pix_fmt"],
                duration, aspect_label, audio_info)
    logger.info("Source fingerprint (computation cache key): sha256:%s...; "
                "video bitrate ~%.2f Mbps.", source_sha256[:16], video_bitrate / 1e6)
    return source
