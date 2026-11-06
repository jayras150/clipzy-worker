# ---------------------------------------------------------------------------
# Clipzy AI - GPU processing worker image (Vast.ai Serverless compatible)
#
# Build:  docker build -t <registry>/clipzy-worker:latest .
# The image bakes in FFmpeg/NVENC, fonts, the YuNet model and Python deps so
# worker startup never downloads large models. GPU access is granted by the
# host runtime (Vast.ai template / `docker run --gpus all`), not at build time.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLIPZY_FONTS_DIR=/usr/local/share/fonts/clipzy \
    CLIPZY_MODELS_DIR=/opt/clipzy/models \
    CLIPZY_CACHE_DIR=/var/cache/clipzy \
    CLIPZY_WORK_ROOT=/tmp/clipzy/jobs \
    CLIPZY_PIPELINE_LOG_FILE=/var/log/clipzy/pipeline.log

# ---- System layer: ffmpeg (libass + NVENC), Python, deno, fonts ---------- -
# nvidia/cuda runtime images ship the driver stubs; the h264_nvenc encoder is
# part of every upstream ffmpeg build and activates when a real GPU + driver
# are mounted by the host. If the NVIDIA driver version on the host mismatches
# the runtime stubs, ffmpeg silently fails the NVENC probe and the worker
# automatically falls back to libx264 (utils/ffmpeg.choose_encoder).
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        ffmpeg fonts-roboto fonts-lato fontconfig ca-certificates \
        python3 python3-pip curl jq bc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Deno runtime: yt-dlp's --js-runtimes deno path for the YouTube EJS solver.
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh -s -- -y && \
    ln -sf /usr/local/.deno/bin/deno /usr/local/bin/deno || \
    echo "WARN: deno install failed; yt-dlp will fall back to other JS runtimes."

# ---- Python layer ----------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install --no-deps vastai-sdk || \
    echo "WARN: vastai-sdk not installable at build time; install it in the template."

# ---- Application layer ------------------------------------------------------
COPY . /app/

# ---- Resources: preset fonts + YuNet model (baked in, validated) -----------
# Fonts from google/fonts GitHub (same list as services/resources.py); the
# runtime re-validates and only fetches what is missing.
RUN python3 - <<'PY'
from pathlib import Path
import urllib.request

FONTS = [
    ("montserrat", "Montserrat%5Bwght%5D.ttf"),
    ("anton", "Anton-Regular.ttf"),
    ("oswald", "Oswald%5Bwght%5D.ttf"),
    ("orbitron", "Orbitron%5Bwght%5D.ttf"),
    ("permanentmarker", "PermanentMarker-Regular.ttf"),
    ("bungee", "Bungee-Regular.ttf"),
    ("sharetechmono", "ShareTechMono-Regular.ttf"),
    ("poppins", "Poppins-SemiBold.ttf"),
    ("playfairdisplay", "PlayfairDisplay%5Bwght%5D.ttf"),
    ("quicksand", "Quicksand%5Bwght%5D.ttf"),
    ("inter", "Inter%5Bopsz%2Cwght%5D.ttf"),
]
BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"
fonts_dir = Path("/usr/local/share/fonts/clipzy")
fonts_dir.mkdir(parents=True, exist_ok=True)
for family, encoded in FONTS:
    target = fonts_dir / encoded
    if target.exists() and target.stat().st_size > 10000:
        continue
    try:
        urllib.request.urlretrieve(f"{BASE}/{family}/{encoded}", target)
        print("fetched", family)
    except Exception as exc:
        print("WARN: font fetch failed:", family, exc)
PY

RUN python3 - <<'PY'
from pathlib import Path
import urllib.request

models_dir = Path("/opt/clipzy/models")
models_dir.mkdir(parents=True, exist_ok=True)
url = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
       "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
target = models_dir / "yunet.onnx"
try:
    urllib.request.urlretrieve(url, target)
    assert target.stat().st_size > 100000, "YuNet model too small (LFS stub?)"
    print("YuNet model baked in:", target.stat().st_size, "bytes")
except Exception as exc:
    print("WARN: YuNet fetch failed (Haar fallback will be used):", exc)
    target.unlink(missing_ok=True)
PY

RUN fc-cache -f && mkdir -p /var/cache/clipzy /var/log/clipzy /tmp/clipzy/jobs && \
    ffmpeg -version | head -n 1 && \
    ffmpeg -hide_banner -filters 2>/dev/null | grep -q ' subtitles ' && \
    echo "libass subtitles filter OK"

EXPOSE 8080

# Vast.ai Serverless: the template's start script runs `python worker.py`
# (PYWORKER_REPO points at this repository). Standalone/local:
#   docker run --gpus all -p 8080:8080 --env-file .env <image> \
#       python worker.py --standalone
CMD ["python3", "worker.py"]
