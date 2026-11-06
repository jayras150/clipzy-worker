# Clipzy AI — GPU Processing Worker

A standalone, production-ready Python application that replaces the GitHub
Actions workflow `.github/workflows/process_clip.yml` with a Vast.ai
Serverless-compatible GPU worker. **100% feature parity** with the original
pipeline — every algorithm, prompt, subtitle preset, validation rule and
output artifact was ported, not simplified (see
[docs/MIGRATION_CHECKLIST.md](docs/MIGRATION_CHECKLIST.md) for the step-by-step
mapping).

```
clipzy-worker/
├── Dockerfile                  CUDA + FFmpeg/NVENC + fonts + YuNet baked in
├── requirements.txt
├── .env.example                every credential/setting (never hardcode keys)
├── worker.py                   Vast.ai PyWorker entrypoint (+ --standalone mode)
├── config.py                   frozen model constants, worker settings
├── schemas/                    job request/response validation + documents
├── engines/                    transcription, curation (GPT-6 Luna), vision, faces
├── processors/                 downloader, probe, scenes, geometry, subtitles, render
├── cache/                      persistent computation-cache manager (SHA256-keyed)
├── storage/                    Cloudflare R2 client + job workspace isolation
├── services/                   job manager, pipeline HTTP server, QC, manifest, fonts
├── pipeline/                   the sequential pipeline orchestrator
├── utils/                      ffmpeg/NVENC helpers, GPU detection, logging, validation
├── clients/submit_job.py       example Vast.ai SDK client (session pattern)
├── tests/                      105 automated tests incl. real-render e2e
└── docs/                       API, deployment, testing, migration checklist
```

## What it does

For each job the worker:

1. validates and normalizes the JSON request,
2. downloads the YouTube source (yt-dlp with the EJS solver, cookie-aware
   player-client ladder, FFprobe-verified),
3. fingerprints the source (SHA256) and enforces the 90-minute limit,
4. transcribes speech with `openai/whisper-large-v3-turbo` via OpenRouter
   (8-minute chunks, 2 s overlap, word-level timestamps),
5. extracts keyframes + scene cuts and scans faces (YuNet, Haar fallback),
6. curates 1–8 viral clips with `openai/gpt-6-luna` (multimodal: frames +
   transcript; three-attempt ladder with truncated-JSON salvage),
7. tracks the primary face inside each clip (IoU tracker), reconciles AI zoom
   intents and the output geometry (manual pinch-zoom window → auto zoom →
   aspect crop/pad → final scale; subtitles burned last so they never zoom),
8. renders H.264/AAC clips — **NVENC when a GPU is present, libx264 CRF 18
   otherwise (identical output contract)** — with optional watermark overlay,
9. runs pixel-level quality control (zoom presence and subtitle presence are
   verified on real exported frames; failures fail the job),
10. uploads clips + Android manifest + status.json (+ debug artifacts on
    failure) to Cloudflare R2, and
11. cleans up its isolated workspace.

Progress is reported through the job record (`/status`), intermediate
`status.json` uploads to R2, and structured logs.

## Quick start (local, no Vast)

```bash
cp .env.example .env          # fill OPENROUTER_API_KEY + R2_* credentials
pip install -r requirements.txt
python worker.py --standalone # pipeline server on 0.0.0.0:8080

curl -X POST localhost:8080/run -H 'Content-Type: application/json' -d '{
  "job_id": "clipzy_001",
  "youtube_url": "https://www.youtube.com/watch?v=example",
  "target_duration": "30-60",
  "subtitle_style": "VIRAL",
  "subtitle_language": "",
  "aspect_ratio": "9:16",
  "preserve_source_quality": true,
  "output_quality": "original",
  "manual_transform": null,
  "watermark_base64": "",
  "watermark_url": "",
  "watermark_position": "bottom_right",
  "watermark_size": "medium"
}'
curl -X POST localhost:8080/status -d '{"job_id": "clipzy_001"}'
```

## Vast.ai Serverless

`worker.py` implements the official PyWorker interface (`vastai` SDK:
`WorkerConfig` / `HandlerConfig` / `BenchmarkConfig` / `LogActionConfig`).
Point `PYWORKER_REPO` at this repository; long jobs use the documented
session pattern (`/run` → poll `/status`, `on_close_route="/cancel"`).
Full instructions: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/API.md](docs/API.md) | job request/response formats, routes, status contract |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, Vast.ai serverless setup, GPU notes |
| [docs/TESTING.md](docs/TESTING.md) | how to run the 105-test suite, what is mocked |
| [docs/MIGRATION_CHECKLIST.md](docs/MIGRATION_CHECKLIST.md) | workflow-step → module mapping, GH-Actions replacements |
