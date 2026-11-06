# Deployment Guide

## 1. Build the Docker image

```bash
docker build -t <registry>/clipzy-worker:latest .
docker push <registry>/clipzy-worker:latest
```

The image (based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`) contains:

* FFmpeg with the libass `subtitles` filter **and** the `h264_nvenc` encoder
  (activated automatically when the host mounts a GPU + driver; libx264 CRF 18
  is the guaranteed CPU fallback),
* Deno (for yt-dlp's `--js-runtimes deno` EJS challenge solver),
* all 11 preset fonts + the YuNet 2023-mar ONNX model — **baked in at build
  time, never downloaded at worker startup**,
* the Python dependencies (OpenCV headless, numpy, requests, `yt-dlp[default]`,
  aiohttp, boto3, vastai-sdk).

GPU extras (`faster-whisper`, `torch`) are deliberately **not** in the base
image; see §5.

## 2. Vast.ai Serverless deployment

`worker.py` implements the official PyWorker interface documented at
<https://docs.vast.ai/guides/serverless/worker-py>.

1. Push this repository to a public Git repo (GitHub is fine — it is only
   source hosting; **no GitHub Actions are involved at runtime**).
2. Create a Serverless Template whose start script uses the PyWorker
   convention and set:
   * `PYWORKER_REPO` = your repository URL
   * `PYWORKER_REF` = branch/tag/commit (optional)
   * `DOCKER_IMAGE` / image override = the image built in §1 (or let the
     template build from the repo's Dockerfile)
3. The start script runs `python worker.py` (the image's default CMD), which:
   * provisions fonts + YuNet (already baked in; re-validated, never
     re-downloaded when valid),
   * starts the internal pipeline server on `127.0.0.1:8080` (the PyWorker's
     "model server") and writes `Application startup complete.` into
     `/var/log/clipzy/pipeline.log` — the `LogActionConfig.on_load` line,
   * starts the PyWorker HTTP server (`WORKER_PORT`, default 3000) with the
     routes `/run`, `/status`, `/cancel`, `/probe` and the benchmark handler
     on `/probe`.
4. Create an Endpoint + Worker Group from the template. Suggested worker
   group settings:
   * GPU: any CUDA GPU ≥ 8 GB VRAM (the pipeline mostly needs NVENC + OpenCV;
     a mid-range card is plenty),
   * Disk ≥ 20 GB (source videos up to 90 minutes + cache),
   * `CLIPZY_MAX_CONCURRENT_JOBS=1` (one heavy pipeline per GPU; Vast
     sessions/FIFO queue the rest).
5. Set the environment variables from `.env.example` in the template
   (`OPENROUTER_API_KEY`, R2 credentials, optional `YT_COOKIES`).

### Cache persistence

Computation caches (transcript/curation/vision/face results keyed by the
source SHA256) live in `CLIPZY_CACHE_DIR` (`/var/cache/clipzy`). Mount a Vast
volume at `/var/cache/clipzy` in the worker group so caches survive worker
restarts and are shared across jobs on the same worker. Cache entries always
re-validate (source hash + model + parameters + pipeline version + TTL), so a
cold or stale cache is safe.

### Long-job integration

Jobs run 10–90 minutes; the Vast SDK per-request timeout is 600 s. Use the
session pattern from `docs/API.md` (POST `/run` once, poll `/status`,
`on_close_route="/cancel"`). The worker also pushes intermediate
`status/<job_id>.json` objects to R2, so the Android app can keep polling the
same URL it used with the GitHub Actions pipeline.

## 3. Standalone deployment (any Docker host / bare metal)

```bash
docker run --gpus all -p 8080:8080 --env-file .env \
    -v clipzy-cache:/var/cache/clipzy \
    <registry>/clipzy-worker:latest python worker.py --standalone
```

`--standalone` runs the pipeline HTTP server directly on `0.0.0.0:8080`
(no vastai SDK needed). Put your own auth in front of it (reverse proxy /
private network) — the standalone server itself does not authenticate.

## 4. GPU behaviour

| Capability | Detection | Behaviour |
| --- | --- | --- |
| GPU presence | `nvidia-smi`, optional torch | reported by `/probe`; used for logging + local whisper device choice |
| NVENC encoding | real encode probe (not just `-encoders`) | `h264_nvenc` VBR cq=18; **automatic libx264 retry on any NVENC failure**; `CLIPZY_ALLOW_NVENC=false` forces CPU |
| GPU VRAM monitoring | `nvidia-smi` | exposed via `/probe`; logs during jobs |
| faster-whisper (optional) | `CLIPZY_LOCAL_WHISPER=1` | local CUDA transcription replaces only the STT *transport*; merge/cache logic and the frozen model-name check are identical |

Not every FFmpeg operation can run on the GPU (filters, remuxes, subtitle
burn are CPU anyway) — NVENC is applied only to the final H.264 encode, which
is where a GPU actually helps. CPU fallbacks exist at every level, so a
CPU-only host still produces identical H.264/AAC output.

## 5. Optional: local faster-whisper

```dockerfile
# append to the Dockerfile for GPU transcription without OpenRouter STT
RUN python3 -m pip install faster-whisper
ENV CLIPZY_LOCAL_WHISPER=true
```

Default remains `false`: the canonical pipeline transcribes through
OpenRouter `openai/whisper-large-v3-turbo` exactly as the workflow did, and
the engine still refuses any other *configured* STT model name.

## 6. Operations

* **Logs**: stderr + `/var/log/clipzy/pipeline.log` (watched by the PyWorker
  `LogActionConfig`; `Traceback` lines mark the worker errored → Vast restarts
  it). Per-job logs are kept in the job workspace (deleted with it).
* **Debug artifacts**: on failure the raw GPT-6 Luna attempts (HTTP status,
  finish reason, token usage, response excerpts — never secrets), render plan
  and hints are uploaded to `outputs/<job_id>/debug/`.
* **Job timeout**: `CLIPZY_JOB_TIMEOUT_SECONDS` (default 5400 s = the
  workflow's 90-minute limit).
* **Scaling**: endpoint `max_workers`/`min_workers`/`target_util` as usual;
  the workload metric is a constant per request (job cost is dominated by the
  background pipeline, and sessions hold the worker while a job runs).
