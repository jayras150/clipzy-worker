"""Branding watermark logo preparation (base64 or URL), ported from PY_WM."""

from __future__ import annotations

import base64
import urllib.request
from pathlib import Path

from utils.logging_setup import get_logger

logger = get_logger("watermark")


def prepare_watermark(params: dict, workdir: Path) -> Path | None:
    """Resolve the watermark PNG into work/watermark_logo.png (or None)."""
    wm_b64 = (params.get("watermark_base64") or "").strip()
    wm_url = (params.get("watermark_url") or "").strip()
    target = Path(workdir) / "watermark_logo.png"

    if wm_b64:
        logger.info("Decoding Base64 watermark logo...")
        try:
            if wm_b64.startswith("data:"):
                wm_b64 = wm_b64.split(",", 1)[-1]
            data = base64.b64decode(wm_b64)
            target.write_bytes(data)
            logger.info("Successfully decoded Base64 watermark_logo.png (%d bytes) "
                        "for FFmpeg overlay.", len(data))
            return target
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to decode Base64 watermark logo: %s; "
                           "skipping watermark overlay.", exc)
            return None

    if wm_url:
        logger.info("Downloading branding watermark logo from %s...", wm_url)
        try:
            with urllib.request.urlopen(wm_url, timeout=60) as response:
                data = response.read()
            if data:
                target.write_bytes(data)
                logger.info("Successfully downloaded watermark_logo.png for FFmpeg overlay.")
                return target
            logger.warning("downloaded watermark file is empty; skipping watermark overlay.")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to download watermark logo: %s; skipping watermark overlay.", exc)
            return None

    logger.info("No watermark_base64 or watermark_url provided: skipping watermark overlay.")
    return None
