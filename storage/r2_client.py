"""Cloudflare R2 (S3-compatible) client.

Replaces the workflow's `aws s3 cp` CLI calls with boto3 while preserving the
exact object layout:

  outputs/<job_id>/clip_N.mp4        every rendered clip
  outputs/<job_id>.mp4               first clip (primary download)
  outputs/<job_id>_meta.json         manifest (Android legacy location)
  outputs/<job_id>/manifest.json     manifest (folder location)
  status/<job_id>.json               status.json
  outputs/<job_id>/debug/...         debug artifacts on failure

Adds: HEAD verification of every uploaded object and bounded retries with
exponential backoff. All credentials come from environment variables.
"""

from __future__ import annotations

import io
import json
import mimetypes
import time
from pathlib import Path

from utils.logging_setup import get_logger

logger = get_logger("r2")


class R2Error(RuntimeError):
    pass


class R2Client:
    """Thin S3-compatible client for Cloudflare R2."""

    def __init__(self, endpoint_url: str, access_key_id: str, secret_access_key: str,
                 bucket_name: str, public_domain: str, region: str = "auto") -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket_name
        self.public_domain = public_domain.rstrip("/")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region or "auto",
            config=Config(retries={"max_attempts": 3, "mode": "standard"},
                          connect_timeout=30, read_timeout=300,
                          signature_version="s3v4"),
        )

    # ------------------------------------------------------------------
    def _put_with_retry(self, key: str, body, content_type: str, attempts: int = 4) -> None:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                                   ContentType=content_type)
                return
            except Exception as exc:  # noqa: BLE001 - boto3 raises many types
                last_error = exc
                delay = min(30, 2 ** attempt)
                logger.warning("R2 upload %s attempt %d/%d failed: %s; retrying in %ss",
                               key, attempt, attempts, exc, delay)
                if attempt < attempts:
                    time.sleep(delay)
        raise R2Error(f"R2 upload failed for {key} after {attempts} attempts: {last_error}")

    def _verify_object(self, key: str, size: int) -> None:
        try:
            head = self.s3.head_object(Bucket=self.bucket, Key=key)
            remote = int(head.get("ContentLength") or 0)
        except Exception as exc:  # noqa: BLE001
            raise R2Error(f"R2 verification failed for {key}: {exc}") from exc
        if remote != size:
            raise R2Error(f"R2 verification failed for {key}: remote size {remote} != {size}")

    # ------------------------------------------------------------------
    def upload_file(self, local_path: str | Path, key: str, content_type: str | None = None) -> str:
        path = Path(local_path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise R2Error(f"cannot upload missing/empty file: {path}")
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            data = fh.read()
        self._put_with_retry(key, data, ctype)
        self._verify_object(key, len(data))
        return self.public_url(key)

    def upload_json(self, document: dict, key: str) -> str:
        body = json.dumps(document, ensure_ascii=False, indent=2)
        self._put_with_retry(key, io.BytesIO(body.encode("utf-8")), "application/json")
        self._verify_object(key, len(body.encode("utf-8")))
        return self.public_url(key)

    def upload_directory(self, local_dir: str | Path, key_prefix: str,
                         content_type: str = "application/octet-stream") -> list[str]:
        urls = []
        for path in sorted(Path(local_dir).iterdir()):
            if path.is_file():
                urls.append(self.upload_file(path, f"{key_prefix}/{path.name}", content_type))
        return urls

    def public_url(self, key: str) -> str:
        return f"{self.public_domain}/{key.lstrip('/')}"

    # ------------------------------------------------------------------
    def upload_job_outputs(self, job_id: str, clips_dir: str | Path,
                           manifest_document: list) -> dict:
        """The workflow's upload step, verbatim object layout."""
        clips_dir = Path(clips_dir)
        import re

        def clip_number(path: Path) -> int:
            match = re.fullmatch(r"clip_(\d+)\.mp4", path.name, flags=re.IGNORECASE)
            return int(match.group(1)) if match else 0

        clip_files = sorted(
            (p for p in clips_dir.glob("clip_*.mp4") if clip_number(p) > 0),
            key=clip_number,
        )
        if not clip_files:
            raise R2Error("no rendered clip found for primary download.")

        urls: dict[str, str] = {}
        for clip_path in clip_files:
            index = clip_number(clip_path)
            object_key = f"outputs/{job_id}/{clip_path.name}"
            urls[f"clip_{index}"] = self.upload_file(
                clip_path, object_key, "video/mp4"
            )
        urls["primary"] = self.upload_file(
            clip_files[0], f"outputs/{job_id}.mp4", "video/mp4"
        )
        manifest_json = json.dumps(manifest_document, ensure_ascii=False, indent=2)
        self._put_with_retry(f"outputs/{job_id}_meta.json", io.BytesIO(manifest_json.encode("utf-8")),
                             "application/json")
        self._verify_object(f"outputs/{job_id}_meta.json", len(manifest_json.encode("utf-8")))
        self._put_with_retry(f"outputs/{job_id}/manifest.json", io.BytesIO(manifest_json.encode("utf-8")),
                             "application/json")
        self._verify_object(f"outputs/{job_id}/manifest.json", len(manifest_json.encode("utf-8")))
        urls["manifest"] = self.public_url(f"outputs/{job_id}/manifest.json")
        return urls

    def upload_status(self, job_id: str, status_document: dict) -> str:
        return self.upload_json(status_document, f"status/{job_id}.json")

    def upload_debug(self, job_id: str, debug_dir: str | Path, extra_files: list) -> list[str]:
        """Debug artifacts (replaces GitHub Actions upload-artifact)."""
        urls = []
        debug_dir = Path(debug_dir)
        if debug_dir.is_dir():
            urls.extend(self.upload_directory(debug_dir, f"outputs/{job_id}/debug", "text/plain"))
        for path in extra_files:
            path = Path(path)
            if path.is_file():
                urls.append(self.upload_file(path, f"outputs/{job_id}/debug/{path.name}",
                                             "application/json"))
        return urls

