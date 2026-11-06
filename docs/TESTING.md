# Testing Guide

## Running the suite

```bash
cd clipzy-worker
pip install -r requirements.txt
pip install pytest pytest-asyncio
python -m pytest            # 105 tests, ~25 s
```

Requirements: `ffmpeg`/`ffprobe` on PATH (the render/e2e tests run real
encodes), Python 3.10+. No network access is needed — see "What is real vs
mocked" below. No GPU is needed: NVENC probes report unavailable and the CPU
(libx264) path is exercised, which is exactly the documented fallback.

## What is real vs mocked

| Stage | In tests | Rationale |
| --- | --- | --- |
| Validation / normalization / schemas | real | pure logic |
| Computation cache (save/load/TTL/integrity) | real | pure logic |
| GPT client protocol (retries, adaptation, JSON repair, schema validation) | real, `requests.post` mocked | protocol logic must not depend on the API |
| Source probing (FFprobe, SHA256, rotation, limits) | **real FFmpeg** on a synthetic video | |
| Scene detection / keyframe extraction | **real FFmpeg** (hard-cut video built with a concat filter) | |
| Face detection (YuNet/Haar loader, IoU tracker, histogram, coarse scan) | real; YuNet download mocked → Haar path | Haar ships with OpenCV |
| Transcription merge (chunking, overlap dedupe, regrouping, cache, model check) | real merge logic; STT endpoint mocked | |
| Curation + refinement (prompts, 3-attempt ladder, salvage, hooks, dedupe) | real logic; chat endpoint mocked with canned valid/truncated responses | |
| Geometry / zoom / subtitle presets | real | pure logic over real transcripts |
| Rendering (unified transform chain, ASS burn, watermark) | **real FFmpeg** renders 640x360 clips | |
| Quality control (codec/fps/duration, black frames, zoom + subtitle presence on exported frames) | **real FFmpeg + OpenCV** | the QC reference renders actually run |
| R2 upload (layout, retries, verification) | real client; boto3 transport mocked | |
| Pipeline server (`/run /status /cancel /probe /health`) | real aiohttp server + job manager threads | |
| End-to-end pipeline | **full run on a synthetic video** (audio chunking → transcript → keyframes → faces → curation → tracking → zoom → ASS → render → QC → manifest → uploads) with the three external boundaries mocked (download, OpenRouter, R2) | proves the assembled pipeline, not just the parts |

The e2e test asserts the complete contract: job response shape, Android
manifest, R2 object layout, burned subtitles **verified at the pixel level**,
AI zoom **verified on exported frames**, workspace cleanup on success,
failure and cancellation.

## Test files

| File | Coverage |
| --- | --- |
| `test_validation.py` | request normalization: defaults, aliases, every rejection, manual-transform clamping |
| `test_cache_manager.py` | cache integrity, TTL, hash scoping |
| `test_gpt_client.py` | retry ladder, parameter adaptation, fatal statuses, truncated-JSON salvage, curation schema, debug artifacts |
| `test_source_probe.py` | probe fields, SHA256 stability/sensitivity, 90-min limit, hard-cut detection, keyframes + cache hit |
| `test_face_detection.py` | IoU tracker identity/aging, histogram cuts, Haar fallback, coarse scan caching |
| `test_geometry_zoom.py` | output-quality policy (no silent upscaling), crop/pad decision, manual window, zoom merge/split/drop, caption band |
| `test_subtitle_generator.py` | all 17 presets, ASS header/style/events, karaoke vs segment timing, top-band alignment |
| `test_transcription.py` | chunk plan, merge, overlap dedupe, word regrouping, cache hit, model refusal, transcript sampling |
| `test_r2_client.py` | object layout, retries, size verification, debug uploads |
| `test_e2e_pipeline.py` | complete pipeline: success, manual transform + aspect, failure + debug upload + cleanup, cancellation |
| `test_pipeline_server.py` | HTTP routes, 422/429/409/404, cancel, envelope payload wrapping, probe/health |
| `test_worker_misc.py` | downloader format/client policy, cookie 0600, encoder choice (CPU/NVENC), GPU detection, workspace isolation, progress stages, PyWorker model-server thread + readiness log line |

## With real credentials

`tests/` never requires credentials. To validate a deployment end-to-end with
the real services, use the example client against a running worker:

```bash
VAST_API_KEY=... python clients/submit_job.py \
    --endpoint clipzy-worker \
    --url https://www.youtube.com/watch?v=<id> \
    --job-id smoke_001 --style VIRAL
```

and verify the returned `download_url`/`manifest_url` objects in your R2
bucket.
