"""YouTube source downloader.

Direct port of the workflow's "Download source video" step. Preserved exactly:
* format-selection policy (height-only selectors, `bv*[height<=N]+ba/bv*+ba/b`,
  2160 cap for original, 1080 cap otherwise, merge to mp4);
* player-client ladders (cookie-aware primary/fallback sets verified against
  yt-dlp's extractor: no tv_embedded, no cookie-incompatible clients in the
  authenticated path);
* cookie handling written to a 0600 file, never logged;
* --check-formats with NO positional argument;
* verification by FFprobe of the file on disk (never yt-dlp's exit code),
  container normalization by remux, storyboard guard and the loud low-res
  warning;
* the diagnostic dump on real failure (no secrets).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("downloader")


class DownloadError(RuntimeError):
    pass


def _format_selector(output_quality: str) -> str:
    if (output_quality or "original") == "original":
        return "bv*[height<=2160]+ba/bv*+ba/b"
    return "bv*[height<=1080]+ba/bv*+ba/b"


def _client_sets(have_cookies: bool) -> tuple[str, str]:
    if have_cookies:
        return ("web_embedded,tv_samsung,web", "tv,tv_samsung,web_embedded,web,web_safari,mweb")
    return ("tv_samsung,visionos,web", "visionos,tv_samsung,android,ios,web")


def _write_cookies(cookie_file: Path, cookies: str) -> list[str]:
    if cookies:
        cookie_file.write_text(f"{cookies}\n", encoding="utf-8")
        os.chmod(cookie_file, 0o600)
        logger.info("YouTube cookies: provided (%d bytes); used for main + diagnostic attempts.",
                    len(cookies))
        return ["--cookies", str(cookie_file)]
    logger.warning("YouTube cookies: NOT provided (anonymous). "
                   "Bot-detection errors will be unrecoverable.")
    return []


def _attempt_download(url: str, clients: str, log_path: Path, video_fmt: str,
                      cookie_flags: list[str], workdir: Path) -> str:
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--js-runtimes", "deno",
        "--extractor-args", f"youtube:player_client={clients}",
        "-f", video_fmt,
        "--merge-output-format", "mp4",
        "--check-formats",
        "--retries", "3",
        "--no-warnings",
        "-o", "raw_video.%(ext)s",
        *cookie_flags,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(workdir), timeout=3600)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(output, encoding="utf-8")

    selected = "(unknown)"
    match = re.search(r"Downloading [0-9]+ format\(s\): (\S+)", output)
    if match:
        selected = match.group(1)
    logger.info("yt-dlp selected format(s): %s", selected)
    if re.search(r"only images are available|storyboard", output, re.IGNORECASE):
        logger.warning("client set [%s] exposed storyboard/image formats only - "
                       "no real video streams.", clients)
    return selected


def _probe_ints(probe: dict) -> tuple[int, int, float, int]:
    width = next((int(s.get("width") or 0) for s in probe.get("streams", [])
                  if s.get("codec_type") == "video"), 0)
    height = next((int(s.get("height") or 0) for s in probe.get("streams", [])
                   if s.get("codec_type") == "video"), 0)
    try:
        duration = round(float(probe.get("format", {}).get("duration") or 0), 3)
    except (TypeError, ValueError):
        duration = 0.0
    return width, height, duration, 0


def _verify_download(workdir: Path) -> tuple[bool, dict]:
    """Trust the FILE on disk, validated by FFprobe — never yt-dlp's exit code."""
    raw_files = sorted(workdir.glob("raw_video.*"))
    if not raw_files:
        logger.warning("VERIFY: no raw_video.* file was produced.")
        return False, {}

    if not (len(raw_files) == 1 and raw_files[0].name == "raw_video.mp4"):
        largest = max(raw_files, key=lambda p: p.stat().st_size)
        if largest.name != "raw_video.mp4":
            logger.info("Normalizing source container: %s -> raw_video.mp4 "
                        "(no re-encode, just remux).", largest.name)
            proc = ff.run([ff.ffmpeg_bin(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                           "-i", str(largest), "-c", "copy", "-map", "0", "-map_metadata", "0",
                           str(workdir / "raw_video.mp4")])
            if proc.returncode != 0:
                logger.warning("remux normalization failed: %s", proc.stderr[-500:])
            largest.unlink(missing_ok=True)

    target = workdir / "raw_video.mp4"
    if not target.is_file() or target.stat().st_size <= 0:
        logger.warning("VERIFY: raw_video.mp4 missing/empty after normalization.")
        return False, {}

    probe = ff.probe_media(target)
    width, height, duration, _ = _probe_ints(probe)
    size = target.stat().st_size
    logger.info("Verified source: %dx%d, duration=%ss, size=%dMB", width, height, duration,
                size // (1024 * 1024))
    if not (width > 0 and height > 0 and size > 10240):
        logger.warning("VERIFY: FFprobe validation failed (width=%d height=%d size=%d).",
                       width, height, size)
        return False, {}
    if duration <= 1.0:
        logger.warning("VERIFY: duration %ss is too short.", duration)
        return False, {}
    return True, {"width": width, "height": height, "duration": duration, "size": size}


def download_source(youtube_url: str, output_quality: str, cookies: str,
                    workdir: Path) -> dict:
    """Download + verify the source. Returns probe info; raises DownloadError."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    video_fmt = _format_selector(output_quality)
    primary_clients, fallback_clients = _client_sets(bool(cookies))
    logger.info("Source format selection (%s): %s", output_quality or "original", video_fmt)
    logger.info("Player clients (primary):  %s", primary_clients)
    logger.info("Player clients (fallback): %s", fallback_clients)

    cookie_file = workdir / "cookies.txt"
    cookie_flags = _write_cookies(cookie_file, cookies)

    attempt_logs: list[Path] = []
    used_clients = "(none)"
    ok, info = False, {}
    for client_set in (primary_clients, fallback_clients):
        logger.info("--- Download attempt with player clients: %s ---", client_set)
        log_path = workdir / f"ytdlp_attempt_{len(attempt_logs) + 1}.log"
        attempt_logs.append(log_path)
        _attempt_download(youtube_url, client_set, log_path, video_fmt, cookie_flags, workdir)
        ok, info = _verify_download(workdir)
        if ok:
            used_clients = client_set
            logger.info("SUCCESS: valid source video obtained with client set [%s].", client_set)
            break
        logger.info("Client set [%s] did not produce a valid video; trying the fallback set...",
                    client_set)

    if not ok:
        _diagnostic_dump(youtube_url, video_fmt, primary_clients, fallback_clients,
                         cookie_flags, attempt_logs, workdir)
        cookie_file.unlink(missing_ok=True)
        for log in attempt_logs:
            log.unlink(missing_ok=True)
        raise DownloadError(
            f"yt-dlp failed to produce a valid video for {youtube_url[:120]} "
            f"(selector {video_fmt}; attempted client sets: {primary_clients}, {fallback_clients}).")

    # Loud warning when the source resolution is below the requested quality
    # (a silent 360p fallback must never go unnoticed).
    if output_quality and output_quality != "original":
        target_short = int(output_quality.replace("p", ""))
        short_side = min(info["width"], info["height"])
        if short_side < target_short:
            logger.warning("source short-side %dpx is BELOW requested %s; "
                           "YouTube returned only muxed/low-res formats for this URL. "
                           "Output will be capped at %dx%d (no upscaling; the render step "
                           "enforces this).", short_side, output_quality, info["width"], info["height"])

    for log in attempt_logs:
        log.unlink(missing_ok=True)
    # Cookie file removed only AFTER success (the retry path may still need it).
    cookie_file.unlink(missing_ok=True)
    logger.info("Download complete: raw_video.mp4 (%(width)dx%(height)d, %(duration)ss) "
                "via client set [%s].", {**info, "clients": used_clients})
    return {**info, "used_clients": used_clients, "format_selector": video_fmt}


def _diagnostic_dump(url: str, video_fmt: str, primary: str, fallback: str,
                     cookie_flags: list[str], attempt_logs: list[Path], workdir: Path) -> None:
    logger.error("----- yt-dlp failed; diagnostic dump follows (no secrets) -----")
    logger.error("URL: %s", url)
    logger.error("Selector: %s", video_fmt)
    logger.error("Primary clients:   %s", primary)
    logger.error("Fallback clients:  %s", fallback)
    try:
        version = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        version = "unknown"
    logger.error("yt-dlp version: %s", version)
    logger.error("Cookies: %s", "yes" if any("--cookies" in f for f in cookie_flags) else "no")
    for index, log in enumerate(attempt_logs, 1):
        logger.error("=== Attempt %d log tail ===\n%s", index, log.read_text(errors="replace")[-2000:])
    for label, clients in (("PRIMARY", primary), ("FALLBACK", fallback)):
        proc = subprocess.run(
            ["yt-dlp", "--no-playlist", "--js-runtimes", "deno",
             "--extractor-args", f"youtube:player_client={clients}", *cookie_flags,
             "--list-formats", url],
            capture_output=True, text=True, timeout=300,
        )
        logger.error("=== Formats visible to %s client set (%s) ===\n%s", label, clients,
                     (proc.stdout or proc.stderr or "")[-2000:])
    logger.error(
        "REASON HINTS:\n"
        "  - Cookies were provided: only cookie-compatible clients were used "
        "(ios/android/visionos are auto-skipped by yt-dlp when cookies are set).\n"
        "  - If the format table above shows only storyboard/image rows, the remaining "
        "clients are PO-token gated for this IP; refresh YT_COOKIES or add a PO-token "
        "provider plugin.\n"
        "  - No cookies: consider setting the YT_COOKIES env var; anonymous worker IPs "
        "are frequently PO-token/bot gated.")
    logger.error("----- end diagnostic dump -----")
