"""Job workspace management: isolation and cleanup.

Every job runs in its own directory under CLIPZY_WORK_ROOT:
  <work_root>/<job_id>__<uuid>/          - isolated workspace (deleted at the end)
  <work_root>/<job_id>__<uuid>/work/     - intermediates (params/source/...)
  <work_root>/<job_id>__<uuid>/output_clips/  - rendered clips

Concurrent jobs can never collide; unrelated jobs never share temporary video
files (the workflow's rule: videos are never cached across jobs).
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from utils.logging_setup import get_logger

logger = get_logger("file_manager")


class JobWorkspace:
    def __init__(self, work_root: Path, job_id: str) -> None:
        self.root = Path(work_root) / f"{job_id}__{uuid.uuid4().hex[:12]}"
        self.work = self.root / "work"
        self.output_clips = self.root / "output_clips"
        self.source = self.root / "raw_video.mp4"
        self.transcript_txt = self.root / "transcript.txt"
        self.manifest = self.root / "clips_manifest.json"

    # -- lifecycle ------------------------------------------------------
    def create(self) -> "JobWorkspace":
        for path in (self.work, self.output_clips):
            path.mkdir(parents=True, exist_ok=True)
        for sub in ("audio", "coarse", "frames", "subs", "debug", "track", "cache"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        return self

    def cleanup(self) -> None:
        """Remove the whole isolated workspace (cookie files, temp videos,
        intermediate frames included). Best effort, never raises."""
        try:
            shutil.rmtree(self.root, ignore_errors=True)
            parent = self.root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError as exc:
            logger.warning("workspace cleanup failed for %s: %s", self.root, exc)

    # -- convenience -----------------------------------------------------
    def path(self, *parts) -> Path:
        return self.work.joinpath(*parts)

    def cache_paths(self, cache_root) -> "JobCachePaths":
        return JobCachePaths(self.work / "cache", cache_root)

    def cookie_file(self) -> Path:
        return self.root / "cookies.txt"


class JobCachePaths:
    """The job-local `work/cache` mirrors the original workflow layout, while
    persistent entries live in the shared cache root (see cache.cache_manager).
    This bridge lets every engine read/write the same relative paths as the
    original scripts with a transparent persistent overlay."""

    def __init__(self, job_cache_dir: Path, persistent_root: Path) -> None:
        self.job_cache_dir = job_cache_dir
        self.persistent_root = Path(persistent_root)

    def resolve(self, relative: str) -> Path:
        return self.job_cache_dir / relative


def free_disk_mb(path: str | Path = "/") -> float:
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 * 1024)
    except OSError:
        return -1.0


def dir_size(path: str | Path) -> tuple[int, int]:
    """(total_bytes, file_count) — used by the cache usage report."""
    total, count = 0, 0
    for root, _dirs, files in os.walk(str(path)):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
                count += 1
            except OSError:
                pass
    return total, count
