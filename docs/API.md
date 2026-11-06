# Clipzy Worker API

The worker exposes the pipeline through the Vast.ai Serverless PyWorker
routes. Clients normally talk to the Vast endpoint, which wraps these routes
with authentication, queueing and sessions; in standalone mode the same
routes are served directly on `:8080`.

The Vast.ai request envelope is
`{"auth_data": {...}, "session_id": ..., "payload": {...}}` — only the
`payload` object reaches the pipeline; the examples below show payloads.

---

## Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/run` | POST | Validate the job and start processing (returns immediately) |
| `/status` | POST | Progress and, when finished, the full job response |
| `/cancel` | POST | Cancel a running job at the next stage boundary |
| `/probe` | POST | Health/benchmark payload (GPU report, no work) |
| `/health` | GET | Liveness probe |

---

## Job request (`POST /run`)

All parameters of the original workflow are preserved, with the original
defaults, validation rules and snake_case/camelCase aliases.

```json
{
  "job_id": "clipzy_001",
  "youtube_url": "https://youtube.com/watch?v=example",
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
}
```

| Field | Type / values | Notes |
| --- | --- | --- |
| `job_id` | `[A-Za-z0-9._-]{1,100}` | default `test_job`; one running job per id per worker |
| `youtube_url` | http(s) URL, no whitespace | required |
| `target_duration` | `45` or `30-60` | clip length window |
| `subtitle_style` | `NONE, VIRAL, PODCAST, IMPACT, NEON, MAGENTA, GOLD, CRIMSON, CYBER, BEAST, MARKER, RETRO, GLITCH, MINIMAL, EMERALD, SUNSET, VIOLET` | `NONE` disables burned-in subtitles |
| `subtitle_language` | BCP-47 (`id`, `en-US`, …) | empty = Whisper auto-detect |
| `aspect_ratio` | `source, 9:16, 16:9, 1:1, 4:5, 4:3, 3:4, 21:9` | `source` preserves the native ratio |
| `preserve_source_quality` | bool | never upscale; keep native resolution/fps |
| `output_quality` | `original, 1080p, 720p, 480p` | short-side budget, capped at source capability |
| `manual_transform` | `{"scale": 1.0-5.0, "offset_x": ±limit, "offset_y": ±limit}` or JSON string or null | pinch-zoom base framing; offsets are source fractions, clamped so the window never leaves the frame |
| `watermark_base64` | data-URI or raw base64 PNG | takes precedence over the URL |
| `watermark_url` | URL of a PNG | fetched once per job |
| `watermark_position` | `top_left, top_right, bottom_left, bottom_right` | |
| `watermark_size` | `small (12%), medium (18%), large (26%)` | fraction of output width |

Invalid payloads return HTTP 422 with the exact validation messages of the
original workflow.

### Response (202)

```json
{"ok": true, "job_id": "clipzy_001", "status": "running",
 "progress": 2, "message": "job accepted; poll /status with this job_id"}
```

HTTP 429 when the worker is at `CLIPZY_MAX_CONCURRENT_JOBS`; HTTP 409 when
the same `job_id` is already running on this worker.

---

## Progress (`POST /status`)

```json
{
  "ok": true, "job_id": "clipzy_001", "status": "running",
  "progress": 58, "stage": "curate",
  "stage_label": "Curating viral clips with GPT-6 Luna",
  "created_at": 1762300000, "started_at": 1762300001, "finished_at": null,
  "response": null, "error": null
}
```

States: `pending → running → completed | failed | cancelled`.
Stage/percent mapping: validate 2, download 12, probe 16, audio 20,
transcribe 34, vision 42, faces 47, curate 58, track 63, zoom 66, subs 70,
refine 74, plan 77, render 92, qc 96, manifest 98, upload 100.

## Job response (inside `/status` when `status=completed`)

```json
{
  "job_id": "clipzy_001",
  "status": "completed",
  "progress": 100,
  "clip_count": 3,
  "download_url": "https://cdn.example.com/outputs/clipzy_001.mp4",
  "manifest_url": "https://cdn.example.com/outputs/clipzy_001/manifest.json",
  "clips": [
    {
      "clip_index": 1, "start_sec": 12.0, "end_sec": 48.5,
      "duration": 36.5, "viral_score": 91, "topic": "...", "hook": "...",
      "title": "...", "description": "...", "tags": ["#investasi", "..."],
      "language": "id",
      "visual_analysis": {"main_subject": "...", "scene": "...",
                          "faces_detected": true},
      "editing": {"aspect_ratio": "9:16", "zoom_enabled": true,
                  "zoom_cuts": [{"start_rel_sec": 2.0, "end_rel_sec": 6.0,
                                 "zoom_level": 1.15, "target": "primary_subject"}],
                  "total_magnification": 1.15, "manual_transform": null},
      "subtitles": {"burned_in": true, "style": "VIRAL", "language": "id",
                    "word_timing_source": "whisper"},
      "subtitle_segments": [{"start": 0.2, "end": 3.1, "text": "..."}],
      "render": {"file": "clip_1.mp4", "width": 1080, "height": 1920,
                 "fps": 30.0, "video_codec": "h264", "audio_codec": "aac",
                 "size_bytes": 18324567}
    }
  ],
  "processing_time": 612.4, "error": null, "video_language": "id",
  "subtitle_style": "VIRAL", "aspect_ratio": "9:16",
  "output_quality": "original", "manual_transform": null,
  "finished_at": 1762300612
}
```

On failure: `status="failed"`, `error` contains the pipeline error, and the
debug artifacts (raw GPT-6 Luna attempts, render plan/hints, manifest) are
uploaded to `outputs/<job_id>/debug/` on R2.

---

## R2 objects (unchanged layout)

| Object | Content |
| --- | --- |
| `outputs/<job_id>/clip_N.mp4` | every rendered clip |
| `outputs/<job_id>.mp4` | first clip (primary download) |
| `outputs/<job_id>_meta.json` | clips manifest (Android legacy location) |
| `outputs/<job_id>/manifest.json` | clips manifest |
| `status/<job_id>.json` | status document the Android app polls |
| `outputs/<job_id>/debug/...` | debug artifacts on failure |

`status/<job_id>.json` keeps the original key set (`job_id, status, file_name,
download_url, manifest_url, clip_count, subtitle_style, aspect_ratio,
output_quality, manual_transform, video_language, clips`) plus `progress` and,
while running, `stage`. Status values: `processing`, `ready`, `failed`.

---

## Client usage (Vast.ai session pattern)

Jobs run 10–90 minutes; the Vast SDK's per-request timeout is 600 s, so use
the documented asynchronous-task pattern:

```python
from vastai import Serverless

async with Serverless() as client:
    endpoint = await client.get_endpoint(name="clipzy-worker")
    session = await endpoint.session(cost=100, lifetime=300,
                                     on_close_route="/cancel")
    try:
        await session.request("/run", payload)
        while True:
            status = (await session.request("/status",
                                            {"job_id": payload["job_id"]},
                                            retry=False))["response"]
            if status["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(10)
    finally:
        await session.close()
```

`on_close_route="/cancel"` guarantees the job is cancelled if the client
disappears. See `clients/submit_job.py` for a complete CLI example.
