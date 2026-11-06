"""Authenticated, best-effort callbacks from the Clipzy worker to Supabase."""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger("clipzy.supabase_callback")


class SupabaseCallback:
    def __init__(self, settings) -> None:
        self.url = getattr(settings, "worker_callback_url", "").strip()
        self.secret = getattr(settings, "worker_callback_secret", "").strip()
        self.enabled = bool(self.url and self.secret)
        self.timeout = 12

        if bool(self.url) != bool(self.secret):
            logger.warning(
                "Supabase callback is partially configured. Both "
                "WORKER_CALLBACK_URL and WORKER_CALLBACK_SECRET are required; callbacks disabled."
            )
        elif not self.enabled:
            logger.warning("Supabase worker callbacks are not configured; callbacks disabled.")

    def send(self, job_id: str, event_type: str, **fields: Any) -> bool:
        if not self.enabled:
            return False

        payload = {"job_id": job_id, "event_type": event_type, **fields}
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-worker-callback-secret": self.secret,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            # Callback failures must not crash or fail the video-processing job.
            logger.warning(
                "Supabase callback failed for job %s (%s): %s",
                job_id, event_type, str(exc)[:500],
            )
            return False
