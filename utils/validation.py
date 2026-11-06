"""Job parameter validation and normalization.

Direct port of the workflow's `PY_NORMALIZE` script: identical defaults,
aliases (snake_case + camelCase), clamps, regexes and error messages. The
manual pinch-zoom transform keeps its exact semantics:
  scale >= 1.0 (1.0 = unzoomed); offset_x/y are window-center offsets in
  SOURCE-fraction units; the visible window is (W/scale, H/scale) centered at
  (0.5+offset_x, 0.5+offset_y), clamped so it never leaves the frame.
"""

from __future__ import annotations

import json
import re

from config import (ASPECT_RATIOS, OUTPUT_QUALITIES, SUBTITLE_STYLES,
                     WATERMARK_POSITIONS, WATERMARK_SIZES)


def _pick(source: dict, *names: str, default=None):
    for name in names:
        if name in source:
            value = source[name]
            if value is not None and str(value).strip() != "":
                return str(value).strip()
    return default


def _normalize_manual_transform(raw, errors: list[str]):
    """Parse + clamp the manual transform; returns the dict or None."""
    if isinstance(raw, str):
        raw = raw.strip() or None
        if raw:
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                errors.append(f"manual_transform is not valid JSON: {str(raw)[:120]}")
                return None
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append("manual_transform must be an object with scale/offset_x/offset_y.")
        return None
    try:
        scale = float(raw.get("scale", 1.0))
        off_x = float(raw.get("offset_x", raw.get("offsetX", 0.0)))
        off_y = float(raw.get("offset_y", raw.get("offsetY", 0.0)))
    except (TypeError, ValueError):
        errors.append("manual_transform scale/offset_x/offset_y must be numbers.")
        return None
    if not (1.0 <= scale <= 5.0):
        clamped = min(5.0, max(1.0, scale))
        print(f"WARN: manual_transform.scale {scale} clamped to {clamped}.")
        scale = clamped
    limit = (1.0 - 1.0 / scale) / 2.0 if scale > 1.0 else 0.0
    for label, value in (("offset_x", off_x), ("offset_y", off_y)):
        if abs(value) > limit + 1e-9:
            print(f"WARN: manual_transform.{label} {value:.4f} clamped to "
                  f"{limit:.4f} so no empty borders appear.")
    off_x = min(limit, max(-limit, off_x))
    off_y = min(limit, max(-limit, off_y))
    if scale > 1.001 or abs(off_x) > 1e-4 or abs(off_y) > 1e-4:
        return {"scale": round(scale, 4), "offset_x": round(off_x, 4), "offset_y": round(off_y, 4)}
    return None


def validate_job_request(payload: dict) -> tuple[dict, list[str]]:
    """Validate a raw job payload. Returns (normalized_params, errors)."""
    errors: list[str] = []

    youtube_url = _pick(payload, "youtube_url", "youtubeUrl", default="")
    job_id = _pick(payload, "job_id", "jobId", default="test_job")
    target_duration = _pick(payload, "target_duration", "targetDuration", default="30-60")
    subtitle_style = _pick(payload, "subtitle_style", "subtitleStyle", default="VIRAL").upper()
    subtitle_language = _pick(payload, "subtitle_language", "subtitleLanguage", default="").lower()
    aspect_ratio = _pick(payload, "aspect_ratio", "aspectRatio", default="source").lower()
    preserve_quality = _pick(payload, "preserve_source_quality", "preserveSourceQuality", default="true").lower()
    output_quality = _pick(payload, "output_quality", "outputQuality", default="original").lower()

    manual_raw = None
    for name in ("manual_transform", "manualTransform"):
        if name in payload and payload[name] is not None:
            manual_raw = payload[name]
            break
    manual_transform = _normalize_manual_transform(manual_raw, errors)

    if not youtube_url:
        errors.append("youtube_url is empty; the Android app must send a video URL.")
    elif not youtube_url.startswith(("http://", "https://")):
        errors.append("youtube_url must start with http:// or https://.")
    elif re.search(r"\s", youtube_url):
        errors.append("youtube_url must not contain whitespace.")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id or ""):
        errors.append("job_id may contain only letters, numbers, dot, underscore and hyphen (max 100 chars).")
    if not re.fullmatch(r"\d+(?:-\d+)?", target_duration or ""):
        errors.append("target_duration must be a number or a range, for example 45 or 30-60.")
    if subtitle_style not in SUBTITLE_STYLES:
        errors.append(f"subtitle_style must be one of: {', '.join(SUBTITLE_STYLES)} (got '{subtitle_style}').")
    if subtitle_language and not re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]{2,8})?", subtitle_language):
        errors.append("subtitle_language must be a BCP-47 style code such as 'id', 'en' or 'en-US' (or empty).")
    if aspect_ratio not in ASPECT_RATIOS:
        errors.append(f"aspect_ratio must be one of: {', '.join(ASPECT_RATIOS)} (got '{aspect_ratio}').")
    if output_quality not in OUTPUT_QUALITIES:
        errors.append(f"output_quality must be one of: {', '.join(OUTPUT_QUALITIES)} (got '{output_quality}').")
    if preserve_quality not in ("true", "false", "1", "0", "yes", "no"):
        errors.append("preserve_source_quality must be true or false.")
    else:
        preserve_quality = "true" if preserve_quality in ("true", "1", "yes") else "false"

    watermark_base64 = _pick(payload, "watermark_base64", "watermarkBase64", default="")
    watermark_url = _pick(payload, "watermark_url", "watermarkUrl", default="")
    watermark_position = _pick(payload, "watermark_position", "watermarkPosition", default="bottom_right").lower()
    if watermark_position not in WATERMARK_POSITIONS:
        watermark_position = "bottom_right"
    watermark_size = _pick(payload, "watermark_size", "watermarkSize", default="medium").lower()
    if watermark_size not in WATERMARK_SIZES:
        watermark_size = "medium"

    params = {
        "youtube_url": youtube_url,
        "job_id": job_id,
        "target_duration": target_duration,
        "subtitle_style": subtitle_style,
        "subtitle_language": subtitle_language,
        "aspect_ratio": aspect_ratio,
        "preserve_source_quality": preserve_quality,
        "output_quality": output_quality,
        "manual_transform": manual_transform,
        "watermark_base64": watermark_base64,
        "watermark_url": watermark_url,
        "watermark_position": watermark_position,
        "watermark_size": watermark_size,
    }
    return params, errors
