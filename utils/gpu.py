"""NVIDIA GPU detection and memory monitoring.

Detection is intentionally dependency-free: nvidia-smi is queried first and
only when missing do we fall back to an optional torch install. FFmpeg NVENC
capability is detected independently in utils.ffmpeg.has_nvenc().
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class GpuInfo:
    available: bool
    name: str
    total_memory_mb: int
    driver: str


def _query_nvidia_smi() -> dict | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        name, total, driver = (part.strip() for part in proc.stdout.splitlines()[0].split(","))
        return {"name": name, "total_memory_mb": int(float(total)), "driver": driver}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _query_torch() -> dict | None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return {
                "name": torch.cuda.get_device_name(0),
                "total_memory_mb": int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)),
                "driver": "torch",
            }
    except Exception:  # noqa: BLE001 - torch is optional
        return None
    return None


@lru_cache(maxsize=1)
def detect_gpu() -> GpuInfo:
    info = _query_nvidia_smi() or _query_torch()
    if info:
        return GpuInfo(True, info["name"], info["total_memory_mb"], info["driver"])
    return GpuInfo(False, "", 0, "")


def gpu_memory_used_mb() -> tuple[int, int] | None:
    """(used_mb, total_mb) via nvidia-smi, or None when unavailable."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        used, total = (int(float(part.strip())) for part in proc.stdout.splitlines()[0].split(","))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_report() -> dict:
    info = detect_gpu()
    report = {"available": info.available, "name": info.name,
              "total_memory_mb": info.total_memory_mb, "driver": info.driver}
    usage = gpu_memory_used_mb()
    if usage:
        report["memory_used_mb"], report["total_memory_mb"] = usage
    return report
