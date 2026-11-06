"""Clipzy Worker configuration.

Every value mirrors a constant from the original `.github/workflows/process_clip.yml`.
OpenRouter model identifiers are FROZEN: the engines refuse to run when they are
substituted (same behavior as the workflow's EXPECTED_*_MODEL checks).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load worker-local environment before WorkerSettings reads os.environ.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# AI model constants (FROZEN - never silently substituted).
# ---------------------------------------------------------------------------
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_MODEL = "openai/gpt-6-luna"
OPENROUTER_STT_MODEL = "openai/whisper-large-v3-turbo"
EXPECTED_CHAT_MODEL = OPENROUTER_CHAT_MODEL
EXPECTED_STT_MODEL = OPENROUTER_STT_MODEL

# Computation-cache versions (must stay in sync with the workflow's cache keys).
TRANSCRIPT_CACHE_VER = "v2"
VISION_CACHE_VER = "v2"
CURATION_PROMPT_VERSION = "p1"

# Output budget for GPT-6 Luna calls (model supports 128k completion tokens).
MAX_CURATION_TOKENS = 32000
MAX_REFINE_TOKENS = 32000

# ---------------------------------------------------------------------------
# Pipeline limits (identical to the workflow).
# ---------------------------------------------------------------------------
MAX_SOURCE_SECONDS = 90 * 60          # 90 minutes
LONG_SOURCE_WARN_SECONDS = 60 * 60    # warn after 60 minutes
CHUNK_SECONDS = 480.0                 # 8-minute speech audio chunks
OVERLAP_SECONDS = 2.0
TRANSCRIPT_TTL_DAYS = 30
VISION_TTL_DAYS = 30
CURATION_TTL_DAYS = 14
MAX_TRANSCRIPT_CHARS = 15000

SUBTITLE_STYLES = [
    "NONE", "VIRAL", "PODCAST", "IMPACT", "NEON", "MAGENTA", "GOLD", "CRIMSON",
    "CYBER", "BEAST", "MARKER", "RETRO", "GLITCH", "MINIMAL", "EMERALD", "SUNSET",
    "VIOLET",
]
ASPECT_RATIOS = ["source", "9:16", "16:9", "1:1", "4:5", "4:3", "3:4", "21:9"]
OUTPUT_QUALITIES = ["original", "1080p", "720p", "480p"]
WATERMARK_POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
WATERMARK_SIZES = ("small", "medium", "large")

# YuNet face detection model (OpenCV Zoo, stored via Git LFS upstream: the
# media.githubusercontent.com endpoint is required, the raw URL only returns a
# 131-byte LFS pointer stub).
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")

# Google Fonts (primary: google/fonts GitHub; fallback: Google Fonts CSS API).
FONT_FAMILIES = [
    ("montserrat", "Montserrat%5Bwght%5D.ttf", "Montserrat:wght@400"),
    ("anton", "Anton-Regular.ttf", "Anton"),
    ("oswald", "Oswald%5Bwght%5D.ttf", "Oswald:wght@400"),
    ("orbitron", "Orbitron%5Bwght%5D.ttf", "Orbitron:wght@400"),
    ("permanentmarker", "PermanentMarker-Regular.ttf", "Permanent+Marker"),
    ("bungee", "Bungee-Regular.ttf", "Bungee"),
    ("sharetechmono", "ShareTechMono-Regular.ttf", "Share+Tech+Mono"),
    ("poppins", "Poppins-SemiBold.ttf", "Poppins:wght@600"),
    ("playfairdisplay", "PlayfairDisplay%5Bwght%5D.ttf", "Playfair+Display:wght@400"),
    ("quicksand", "Quicksand%5Bwght%5D.ttf", "Quicksand:wght@400"),
    ("inter", "Inter%5Bopsz%2Cwght%5D.ttf", "Inter:wght@400"),
]
GOOGLE_FONTS_BASE_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl"


@dataclass
class WorkerSettings:
    """Runtime settings, resolved from environment variables."""

    # --- OpenRouter -----------------------------------------------------
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "").strip())
    openrouter_api_base: str = field(default_factory=lambda: (os.environ.get("OPENROUTER_API_BASE") or OPENROUTER_API_BASE).rstrip("/"))
    openrouter_chat_model: str = field(default_factory=lambda: os.environ.get("OPENROUTER_CHAT_MODEL", "").strip() or OPENROUTER_CHAT_MODEL)
    openrouter_stt_model: str = field(default_factory=lambda: os.environ.get("OPENROUTER_STT_MODEL", "").strip() or OPENROUTER_STT_MODEL)

    # --- YouTube --------------------------------------------------------
    yt_cookies: str = field(default_factory=lambda: os.environ.get("YT_COOKIES", ""))

    # --- Cloudflare R2 --------------------------------------------------
    r2_access_key_id: str = field(default_factory=lambda: os.environ.get("R2_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID", "")))
    r2_secret_access_key: str = field(default_factory=lambda: os.environ.get("R2_SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "")))
    r2_endpoint_url: str = field(default_factory=lambda: os.environ.get("R2_ENDPOINT_URL", ""))
    r2_bucket_name: str = field(default_factory=lambda: os.environ.get("R2_BUCKET_NAME", ""))
    r2_public_domain: str = field(default_factory=lambda: os.environ.get("R2_PUBLIC_DOMAIN", "").rstrip("/"))
    r2_region: str = field(default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "auto"))
    r2_required: bool = field(default_factory=lambda: _env_bool("CLIPZY_R2_REQUIRED", True))

    # --- Supabase worker callback ----------------------------------------
    worker_callback_url: str = field(default_factory=lambda: os.environ.get("WORKER_CALLBACK_URL", "").strip().rstrip("/"))
    worker_callback_secret: str = field(default_factory=lambda: os.environ.get("WORKER_CALLBACK_SECRET", "").strip())
    # --- Storage / cache / resources -------------------------------------
    work_root: Path = field(default_factory=lambda: Path(os.environ.get("CLIPZY_WORK_ROOT", "/tmp/clipzy/jobs")))
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("CLIPZY_CACHE_DIR", "/var/cache/clipzy")))
    fonts_dir: Path = field(default_factory=lambda: Path(os.environ.get("CLIPZY_FONTS_DIR", "/usr/local/share/fonts/clipzy")))
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("CLIPZY_MODELS_DIR", "/opt/clipzy/models")))

    # --- GPU / encoding ---------------------------------------------------
    allow_nvenc: bool = field(default_factory=lambda: _env_bool("CLIPZY_ALLOW_NVENC", True))
    # Optional local GPU transcription with faster-whisper. DEFAULT OFF: the
    # canonical pipeline transcribes via OpenRouter whisper-large-v3-turbo.
    local_whisper: bool = field(default_factory=lambda: _env_bool("CLIPZY_LOCAL_WHISPER", False))
    local_whisper_model: str = field(default_factory=lambda: os.environ.get("CLIPZY_LOCAL_WHISPER_MODEL", "large-v3-turbo"))
    local_whisper_device: str = field(default_factory=lambda: os.environ.get("CLIPZY_LOCAL_WHISPER_DEVICE", "cuda"))
    local_whisper_compute_type: str = field(default_factory=lambda: os.environ.get("CLIPZY_LOCAL_WHISPER_COMPUTE_TYPE", "float16"))

    # --- Worker behaviour --------------------------------------------------
    max_concurrent_jobs: int = field(default_factory=lambda: int(os.environ.get("CLIPZY_MAX_CONCURRENT_JOBS", "1")))
    job_keep_seconds: float = field(default_factory=lambda: float(os.environ.get("CLIPZY_JOB_KEEP_SECONDS", "3600")))
    status_uploads: bool = field(default_factory=lambda: _env_bool("CLIPZY_STATUS_UPLOADS", True))
    upload_debug: bool = field(default_factory=lambda: _env_bool("CLIPZY_UPLOAD_DEBUG", True))
    job_timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("CLIPZY_JOB_TIMEOUT_SECONDS", str(90 * 60))))

    # --- Internal pipeline server (the PyWorker "model server") -----------
    pipeline_host: str = field(default_factory=lambda: os.environ.get("CLIPZY_PIPELINE_HOST", "127.0.0.1"))
    pipeline_port: int = field(default_factory=lambda: int(os.environ.get("CLIPZY_PIPELINE_PORT", "8080")))
    pipeline_log_file: Path = field(default_factory=lambda: Path(os.environ.get("CLIPZY_PIPELINE_LOG_FILE", "/var/log/clipzy/pipeline.log")))

    def ensure_dirs(self) -> None:
        for path in (self.work_root, self.cache_dir, self.models_dir):
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except OSError:
                # Read-only defaults fall back to a user-writable location.
                fallback = Path.home() / ".cache" / "clipzy" / Path(path).name
                fallback.mkdir(parents=True, exist_ok=True)
                if path == self.cache_dir:
                    self.cache_dir = fallback
                elif path == self.models_dir:
                    self.models_dir = fallback

    def r2_configured(self) -> bool:
        return bool(self.r2_access_key_id and self.r2_secret_access_key
                    and self.r2_endpoint_url and self.r2_bucket_name and self.r2_public_domain)


def get_settings() -> WorkerSettings:
    return WorkerSettings()


