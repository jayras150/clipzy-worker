# Migration Checklist — process_clip.yml → clipzy-worker

Source of truth: `.github/workflows/process_clip.yml` (4,807 lines).
Every workflow step was ported; the mapping below is exhaustive.

## 1. Trigger / inputs → HTTP API

| Workflow | Worker |
| --- | --- |
| `repository_dispatch` / `workflow_dispatch` inputs | `POST /run` JSON payload (Vast.ai envelope `payload`) |
| `PY_NORMALIZE` validation & normalization (defaults, aliases, clamps, regexes, error messages) | `utils/validation.py` + `schemas/job_request.py` — ported verbatim |
| `GITHUB_ENV` parameter hand-off | `PipelineContext` in-process object + `work/params.json` |
| `concurrency: clipzy-<job_id>` | one running job per `job_id` per worker (409 on duplicates) + `CLIPZY_MAX_CONCURRENT_JOBS` |
| `timeout-minutes: 90` | `CLIPZY_JOB_TIMEOUT_SECONDS` (default 5400) |

## 2. Environment / caches → native infrastructure

| Workflow | Worker |
| --- | --- |
| `actions/cache` transcript cache (`clipzy-transcript-v2-<sha>-<model>-<lang>`) | `cache/cache_manager.py` + `CachePaths.transcript()` — same key ingredients + embedded TTL + restore-time validation |
| `actions/cache` vision cache (coarse/frames/frames.json/face_context) | `CachePaths` vision paths — same meta validation, missing-file re-extraction |
| `actions/cache` curation cache (`…-p1`) | `CachePaths.curation()` with `CURATION_PROMPT_VERSION="p1"` |
| `actions/cache` YuNet model | baked into the image + `engines/face_detection.ensure_yunet_model()` (download-with-validation fallback preserved, LFS media URL preserved) |
| `actions/cache` fonts | baked into the image + `services/resources.ensure_fonts()` (same Google Fonts list + CSS API fallback) |
| `actions/cache` Python venv / pip cache | Docker image layers |
| `actions/cache` Deno cache | Deno installed in the image |
| `upload-artifact` (debug on failure) | R2 `outputs/<job_id>/debug/` uploads (same file set: `work/debug/*`, render plan/hints, manifest) |
| `GITHUB_TOKEN`, secrets | environment variables (`.env.example`): `OPENROUTER_API_KEY`, `YT_COOKIES`, `R2_*` |

Videos are never cached; every job downloads its own source (unchanged rule).

## 3. Pipeline steps → modules (verbatim algorithm ports)

| Workflow step | Module |
| --- | --- |
| Download source video (format ladder, client sets, cookies 0600, `--check-formats` no-arg, remux normalization, FFprobe verification, low-res warning, diagnostic dump) | `processors/downloader.py` |
| Probe source + duration limits (rotation, fps snapping, aspect label, SHA256 fingerprint, bitrate/audio info) | `processors/source_probe.py` |
| Prepare speech audio chunks (16 kHz mono AAC, 480 s / 2 s overlap) | `engines/transcription.prepare_audio_chunks` |
| Download watermark (base64 with data-URI strip / URL) | `processors/watermark.py` |
| Transcribe with Whisper (429-retry ladder, verbose_json word+segment, 4-way parallel, overlap dedupe, word→segment regroup, transcript.txt sampling, cache) | `engines/transcription.transcribe_source` |
| Extract representative keyframes (coarse grid ≤400, scene ≤40 → 8 frames, 12-frame AI set, end-cap rules, cache) | `engines/vision_engine.py` + `processors/scene_detection.py` |
| Scan for faces coarse (YuNet 0.6/0.3/5000 @ ≤640px, Haar fallback, cache) | `engines/face_detection.py` |
| Curate with GPT-6 Luna (full prompt text, 32k budget, 3-attempt ladder, finish_reason handling, truncated-JSON salvage, schema validation, hook verbatim check, tag sanitize, overlap dedupe, cache) | `engines/viral_detection.curate_clips` + `engines/gpt_client.py` |
| Track faces in clips (2 fps @640px, IoU tracker, histogram cuts, primary = persistence×area×centrality with cut penalty, EMA center samples) | `engines/face_detection.track_faces_in_clips` |
| Reconcile zoom + geometry (manual window, output-quality policy, crop/pad decision, zoom budget z_max, window merge/cap, face-follow vs center steering, crop keypoints, caption band) | `processors/auto_crop.py` + `processors/auto_zoom.py` |
| Prepare subtitle preset fonts (11 families, CSS API fallback, fc-cache) | `services/resources.py` (+ Docker build) |
| Generate styled ASS subtitles (all 17 presets, hex→ASS, karaoke `\k`, blur, `\N` wrapping, cue building/splitting, overlap resolution, top/bottom band) | `processors/subtitle_generator.py` |
| Refine clip metadata with GPT-6 Luna (prompt, verbatim hook verification, tag sanitize, CV facts injection) | `engines/viral_detection.refine_metadata` |
| Build render plan (ramp_window, center_expr, zoom chain, manual crop, aspect crop, pad graph, watermark overlay, QC graph variants, render report) | `processors/video_cutter.py` + `processors/renderer.build_render_plan` |
| Render clips (libx264 CRF 18 preset medium/faster, GOP, faststart, AAC 192k, libass warnings, probes, render report printout) | `processors/renderer.render_clips` (+ NVENC option with libx264 retry) |
| Report cache usage | `pipeline/runner._cache_report` |
| Quality control (codec/pix_fmt/resolution/ratio/fps/duration/audio, decode `-xerror`, blackdetect, zoom presence vs neutral reference, subtitle pixel-isolated presence, metadata sanity, placeholder rejection) | `services/quality_control.py` |
| Build Android manifest | `services/manifest.py` |
| Upload to R2 (clips dir, primary clip, `_meta.json`, folder manifest, status.json) | `storage/r2_client.py` (+ retries + HEAD verification) |
| Debug artifacts on failure | `pipeline/runner.py` finally-block |

## 4. New infrastructure (additions, not substitutions)

* `services/job_manager.py` — isolated, cancellable, bounded job execution.
* `services/pipeline_server.py` + `worker.py` — the Vast.ai PyWorker wiring
  (documented `WorkerConfig`/`HandlerConfig`/`LogActionConfig`/
  `BenchmarkConfig` interface; `/probe` is the benchmark handler).
* `services/progress.py` — stage/percent model + intermediate status uploads.
* GPU layer: NVENC encode with automatic CPU fallback, GPU/VRAM detection,
  optional local faster-whisper (`CLIPZY_LOCAL_WHISPER`, default off).
* `clients/submit_job.py` — Vast.ai SDK session-pattern client.

## 5. Frozen invariants (must never drift)

* `OPENROUTER_API_BASE=https://openrouter.ai/api/v1`
* `OPENROUTER_CHAT_MODEL=openai/gpt-6-luna` (refuse-to-run guard preserved)
* `OPENROUTER_STT_MODEL=openai/whisper-large-v3-turbo` (refuse-to-run guard preserved)
* Output contract: H.264 + yuv420p + AAC 192k, never upscale silently,
  subtitles burned after all geometry.
* status.json key set and R2 object layout (plus additive `progress`/`stage`).

## 6. Verification

* 105 automated tests including a full end-to-end run on a real synthetic
  video with real FFmpeg rendering, ASS subtitle burning, zoom-presence and
  subtitle-presence QC (see `docs/TESTING.md`).
* What genuinely requires live credentials: the OpenRouter calls and a real
  YouTube download (network services; both are exercised through the exact
  same call paths in tests with mocked transports).
