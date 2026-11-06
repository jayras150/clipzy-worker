"""Reusable resource provisioning: subtitle preset fonts + YuNet model.

In the Docker image these are baked in at build time (no per-startup model
downloads); this module verifies them at runtime and re-downloads only what
is missing or invalid (same fetch logic + Google Fonts CSS API fallback as the
workflow, minus sudo/runner-specific behavior).
"""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

from config import FONT_FAMILIES, GOOGLE_FONTS_BASE_URL
from utils.logging_setup import get_logger

logger = get_logger("resources")


def _fetch_url(url: str, timeout: float) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except Exception:  # noqa: BLE001
        return None


def ensure_fonts(fonts_dir: Path) -> int:
    """Make sure every preset font exists; returns the number of TTFs present."""
    fonts_dir = Path(fonts_dir)
    fonts_dir.mkdir(parents=True, exist_ok=True)
    installed = 0
    for _family, encoded, css_family in FONT_FAMILIES:
        target = fonts_dir / encoded
        if target.is_file() and target.stat().st_size > 10_000:
            installed += 1
            continue
        data = _fetch_url(f"{GOOGLE_FONTS_BASE_URL}/{_family}/{encoded}", 90)
        if data is None:
            # Fallback: Google Fonts CSS API static TTF (fonts.gstatic.com).
            css = _fetch_url(f"https://fonts.googleapis.com/css2?family={css_family}", 20)
            url = None
            if css:
                text = css.decode("utf-8", errors="replace")
                for part in text.split("https://"):
                    if part.startswith("fonts.gstatic.com") and ")" in part:
                        url = "https://" + part.split(")")[0]
                        break
            if url:
                data = _fetch_url(url, 60)
        if data and len(data) > 10_000:
            target.write_bytes(data)
            installed += 1
            logger.info("Fetched font family: %s", _family)
        else:
            logger.warning("could not fetch '%s'; libass will use a fallback font for that "
                           "preset.", _family)
    try:
        subprocess.run(["fc-cache", "-f"], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass
    logger.info("Font set ready: %d/%d preset font file(s) present.", installed, len(FONT_FAMILIES))
    return installed
