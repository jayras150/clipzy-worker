#!/usr/bin/env python3
"""Example client: submit a Clipzy job to a Vast.ai Serverless endpoint.

Uses the documented asynchronous-task session pattern
(https://docs.vast.ai/guides/serverless/sessions):

  1. open a session (pins + reserves one GPU worker),
  2. POST /run once - the worker validates the payload and starts the
     pipeline in the background, returning immediately,
  3. poll /status until completed/failed/cancelled,
  4. the session is closed when the client exits (on_close_route="/cancel"
     guarantees the job is cancelled if the client dies).

Usage:
    python clients/submit_job.py --url https://www.youtube.com/watch?v=... \
        --job-id clipzy_001 [--style VIRAL] [--aspect 9:16] [--quality original]

Requires: pip install vastai-sdk  (VAST_API_KEY env var must be set)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_job(endpoint_name: str, payload: dict, poll_interval: float = 10.0) -> dict:
    from vastai import Serverless

    async with Serverless() as client:
        endpoint = await client.get_endpoint(name=endpoint_name)
        # cost mirrors one representative request; lifetime longer than the
        # poll gap; cancel the job automatically if this client disappears.
        session = await endpoint.session(cost=100, lifetime=300,
                                         on_close_route="/cancel")
        try:
            started = await session.request("/run", payload)
            if not started.get("ok"):
                raise RuntimeError(f"/run rejected: {started}")
            print(f"job {payload.get('job_id')} accepted: {started}")
            while True:
                polled = await session.request("/status",
                                               {"job_id": payload["job_id"]},
                                               retry=False)
                status = polled["response"]
                state = status.get("status")
                print(f"[{payload['job_id']}] {state} {status.get('progress', 0)}% "
                      f"- {status.get('stage_label', '')}")
                if state in ("completed", "failed", "cancelled"):
                    return status
                await asyncio.sleep(poll_interval)
        finally:
            await session.close()


def build_payload(args: argparse.Namespace) -> dict:
    manual = None
    if args.manual_transform:
        manual = json.loads(args.manual_transform)
    return {
        "job_id": args.job_id,
        "youtube_url": args.url,
        "target_duration": args.target_duration,
        "subtitle_style": args.style,
        "subtitle_language": args.language,
        "aspect_ratio": args.aspect,
        "preserve_source_quality": "true" if args.preserve_quality else "false",
        "output_quality": args.quality,
        "manual_transform": manual,
        "watermark_base64": args.watermark_base64 or "",
        "watermark_url": args.watermark_url or "",
        "watermark_position": args.watermark_position,
        "watermark_size": args.watermark_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a Clipzy processing job")
    parser.add_argument("--endpoint", default="clipzy-worker",
                        help="Vast.ai serverless endpoint name")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--job-id", default="clipzy_001")
    parser.add_argument("--target-duration", default="30-60")
    parser.add_argument("--style", default="VIRAL",
                        choices=["NONE", "VIRAL", "PODCAST", "IMPACT", "NEON", "MAGENTA",
                                 "GOLD", "CRIMSON", "CYBER", "BEAST", "MARKER", "RETRO",
                                 "GLITCH", "MINIMAL", "EMERALD", "SUNSET", "VIOLET"])
    parser.add_argument("--language", default="", help="BCP-47 hint (empty = auto)")
    parser.add_argument("--aspect", default="source",
                        choices=["source", "9:16", "16:9", "1:1", "4:5", "4:3", "3:4", "21:9"])
    parser.add_argument("--quality", default="original",
                        choices=["original", "1080p", "720p", "480p"])
    parser.add_argument("--no-preserve-quality", dest="preserve_quality",
                        action="store_false")
    parser.add_argument("--manual-transform", default="",
                        help='JSON {"scale":1.0,"offset_x":0.0,"offset_y":0.0}')
    parser.add_argument("--watermark-base64", default="")
    parser.add_argument("--watermark-url", default="")
    parser.add_argument("--watermark-position", default="bottom_right",
                        choices=["top_left", "top_right", "bottom_left", "bottom_right"])
    parser.add_argument("--watermark-size", default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()

    payload = build_payload(args)
    result = asyncio.run(run_job(args.endpoint, payload, args.poll_interval))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
