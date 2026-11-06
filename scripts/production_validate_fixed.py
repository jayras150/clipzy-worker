#!/usr/bin/env python3
"""Safe pre-production checks for Clipzy Worker.

Loads .env when python-dotenv is installed. Never prints credential values.
R2 test writes one uniquely named object under _healthchecks/ and deletes it.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from config import (  # noqa: E402
    OPENROUTER_CHAT_MODEL,
    OPENROUTER_STT_MODEL,
    WorkerSettings,
)

passed = 0
failed = 0
warnings = 0


def report(name: str, ok: bool, detail: str = "", warning: bool = False) -> None:
    global passed, failed, warnings
    if warning:
        warnings += 1
        state = "WARN"
    elif ok:
        passed += 1
        state = "PASS"
    else:
        failed += 1
        state = "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{state}] {name}{suffix}")


def http_json(url: str, headers: dict[str, str], timeout: int = 20):
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as response:
        return response.status, response.read()


def main() -> int:
    settings = WorkerSettings()
    print("\n=== CLIPZY WORKER PRODUCTION VALIDATION ===")
    print(f"Project: {ROOT}")
    print("Credential values are never displayed.\n")

    required = {
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
        "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
        "R2_ENDPOINT_URL": settings.r2_endpoint_url,
        "R2_BUCKET_NAME": settings.r2_bucket_name,
        "R2_PUBLIC_DOMAIN": settings.r2_public_domain,
    }
    for key, value in required.items():
        report(f"Environment: {key}", bool(value), "configured" if value else "missing")

    report("OpenRouter base URL", settings.openrouter_api_base.startswith("https://"),
           settings.openrouter_api_base if settings.openrouter_api_base.startswith("https://") else "must use HTTPS")
    report("Chat model pinned", settings.openrouter_chat_model == OPENROUTER_CHAT_MODEL,
           settings.openrouter_chat_model)
    report("STT model pinned", settings.openrouter_stt_model == OPENROUTER_STT_MODEL,
           settings.openrouter_stt_model)
    report("R2 required in production", settings.r2_required,
           "CLIPZY_R2_REQUIRED must be true")

    # Validate numerical/runtime settings before attempting network calls.
    report("Max concurrent jobs", settings.max_concurrent_jobs >= 1,
           str(settings.max_concurrent_jobs))
    report("Job timeout", settings.job_timeout_seconds > 0,
           f"{settings.job_timeout_seconds:g} seconds")

    # Check writable directories.
    try:
        settings.ensure_dirs()
        for label, folder in (
            ("Work directory", settings.work_root),
            ("Cache directory", settings.cache_dir),
            ("Models directory", settings.models_dir),
        ):
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / f".clipzy-write-test-{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report(f"Filesystem: {label}", True, str(folder))
    except Exception as exc:
        report("Filesystem directories", False, f"{type(exc).__name__}: {exc}")

    # OpenRouter authentication and model visibility (GET only; no billable inference).
    if settings.openrouter_api_key:
        try:
            status, body = http_json(
                f"{settings.openrouter_api_base}/models",
                {"Authorization": f"Bearer {settings.openrouter_api_key}",
                 "HTTP-Referer": "https://clipzy.ai",
                 "X-Title": "Clipzy Worker Production Validation"},
            )
            import json
            payload = json.loads(body.decode("utf-8"))
            ids = {item.get("id") for item in payload.get("data", [])}
            report("OpenRouter API authentication", status == 200, f"HTTP {status}")
            report("OpenRouter chat model visible", settings.openrouter_chat_model in ids,
                   settings.openrouter_chat_model)

            # Audio transcription models are listed through the dedicated
            # transcription-modality catalog, not the default /models list.
            try:
                stt_status, stt_body = http_json(
                    f"{settings.openrouter_api_base}/models?output_modalities=transcription",
                    {"Authorization": f"Bearer {settings.openrouter_api_key}",
                     "HTTP-Referer": "https://clipzy.ai",
                     "X-Title": "Clipzy Worker Production Validation"},
                )
                stt_payload = json.loads(stt_body.decode("utf-8"))
                stt_ids = {
                    item.get("id") for item in stt_payload.get("data", [])
                }
                report(
                    "OpenRouter STT model visible",
                    stt_status == 200 and settings.openrouter_stt_model in stt_ids,
                    settings.openrouter_stt_model,
                )
            except HTTPError as exc:
                report(
                    "OpenRouter STT model visible",
                    False,
                    f"HTTP {exc.code}; transcription catalog request failed",
                )
            except (URLError, TimeoutError) as exc:
                report(
                    "OpenRouter STT model visible",
                    False,
                    f"network error: {exc.reason if isinstance(exc, URLError) else exc}",
                )
            except Exception as exc:
                report(
                    "OpenRouter STT model visible",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
        except HTTPError as exc:
            report("OpenRouter API authentication", False, f"HTTP {exc.code}; check key/account access")
        except (URLError, TimeoutError) as exc:
            report("OpenRouter network", False, str(exc.reason if isinstance(exc, URLError) else exc))
        except Exception as exc:
            report("OpenRouter validation", False, f"{type(exc).__name__}: {exc}")
    else:
        report("OpenRouter connectivity", False, "skipped: API key missing")

    # R2 connectivity, write, verify, and delete using a temporary object.
    if settings.r2_configured():
        key = f"_healthchecks/clipzy-worker-{uuid.uuid4().hex}.txt"
        try:
            from storage.r2_client import R2Client
            client = R2Client(
                settings.r2_endpoint_url, settings.r2_access_key_id,
                settings.r2_secret_access_key, settings.r2_bucket_name,
                settings.r2_public_domain, settings.r2_region,
            )
            client.s3.head_bucket(Bucket=settings.r2_bucket_name)
            report("Cloudflare R2 bucket access", True, settings.r2_bucket_name)
            payload = b"clipzy-worker-production-validation"
            client.s3.put_object(Bucket=settings.r2_bucket_name, Key=key,
                                 Body=payload, ContentType="text/plain")
            report("Cloudflare R2 write", True, "temporary healthcheck object uploaded")
            head = client.s3.head_object(Bucket=settings.r2_bucket_name, Key=key)
            report("Cloudflare R2 read/verify", head.get("ContentLength") == len(payload),
                   f"remote size={head.get('ContentLength')}")
            client.s3.delete_object(Bucket=settings.r2_bucket_name, Key=key)
            report("Cloudflare R2 cleanup", True, "temporary object deleted")
        except Exception as exc:
            report("Cloudflare R2 integration", False,
                   f"{type(exc).__name__}: {exc}; check endpoint, bucket and R2 permissions")
    else:
        report("Cloudflare R2 integration", False, "skipped: one or more R2 settings missing")

    # Local pipeline readiness: expected only after worker/pipeline has started.
    import json
    from urllib.request import Request, urlopen
    health_url = f"http://127.0.0.1:{settings.pipeline_port}/health"
    try:
        req = Request(health_url, method="GET")
        with urlopen(req, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
            report("Internal pipeline backend", response.status == 200 and payload.get("service") == "clipzy-pipeline",
                   f"HTTP {response.status}")
    except Exception:
        report("Internal pipeline backend", False,
               f"not reachable at {health_url}; start worker.py first")

    # The worker has no Supabase/backend URL setting; its internal HTTP API is
    # checked above. External Supabase/Vast control-plane validation is separate.
    if os.getenv("SUPABASE_URL") or os.getenv("CLIPZY_BACKEND_URL"):
        report("External backend URL", True, "URL variable detected; connectivity/auth not tested by this worker check",
               warning=True)
    else:
        report("External Supabase/backend", True,
               "not configured in this worker; validate from ClipzyBackend separately",
               warning=True)

    print("\n=== SUMMARY ===")
    print(f"PASS: {passed} | FAIL: {failed} | WARN: {warnings}")
    if failed:
        print("RESULT: NOT READY FOR PRODUCTION")
        return 1
    print("RESULT: WORKER CHECKS PASSED (review warnings and run Docker/Linux validation next)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
