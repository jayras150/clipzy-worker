"""The internal Clipzy pipeline HTTP server ("model server").

The Vast.ai PyWorker framework (vastai SDK) is an HTTP proxy between the
serverless routing layer and a model server. Clipzy has no external model
server, so this aiohttp application IS the backend the PyWorker forwards to:

  POST /run      - validate + start one processing job (returns immediately)
  POST /status   - job progress / final structured response
  POST /cancel   - cancel a running job
  POST /probe    - lightweight health/benchmark payload (returns instantly)
  GET  /health   - readiness check

The PyWorker wraps these with queueing (FIFO when allow_parallel_requests is
False), workload accounting, signature verification and session management
(see worker.py). The request body arriving here is the `payload` object of
the Vast.ai request envelope.
"""

from __future__ import annotations

import json

from aiohttp import web

from config import WorkerSettings
from schemas.job_request import parse_job_payload
from services.job_manager import COMPLETED, RUNNING, JobManager
from utils import gpu
from utils.logging_setup import get_logger

logger = get_logger("pipeline_server")


def _json_error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)


def build_app(settings: WorkerSettings, job_manager: JobManager) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)  # watermark_base64 payloads

    async def run_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _json_error("request body is not valid JSON.")
        if not isinstance(body, dict):
            return _json_error("request body must be a JSON object.")

        job_request, errors = parse_job_payload(body)
        if job_request is None:
            return _json_error("invalid job parameters: " + "; ".join(errors), status=422)

        if job_manager.active_count() >= settings.max_concurrent_jobs:
            return web.json_response(
                {"ok": False, "error": "worker is at max concurrent jobs"},
                status=429)

        try:
            record = job_manager.submit(job_request)
        except RuntimeError as exc:
            return _json_error(str(exc), status=409)

        return web.json_response({
            "ok": True,
            "job_id": record.job_id,
            "status": record.state,
            "progress": record.progress,
            "message": "job accepted; poll /status with this job_id",
        }, status=202)

    async def status_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _json_error("request body is not valid JSON.")
        job_id = str((body or {}).get("job_id", ""))
        record = job_manager.get(job_id)
        if record is None:
            return _json_error(f"unknown job_id '{job_id}'", status=404)
        snapshot = record.snapshot()
        if record.state == COMPLETED:
            return web.json_response({"ok": True, **snapshot})
        return web.json_response({"ok": True, **snapshot})

    async def cancel_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _json_error("request body is not valid JSON.")
        job_id = str((body or {}).get("job_id", ""))
        cancelled = job_manager.cancel(job_id)
        record = job_manager.get(job_id)
        if record is None:
            return _json_error(f"unknown job_id '{job_id}'", status=404)
        return web.json_response({"ok": True, "job_id": job_id,
                                  "cancelled": cancelled,
                                  "status": record.state})

    async def probe_handler(request: web.Request) -> web.Response:
        """Benchmark/health payload: never starts a real job."""
        return web.json_response({
            "ok": True,
            "service": "clipzy-pipeline",
            "gpu": gpu.gpu_report(),
            "active_jobs": job_manager.active_count(),
            "max_concurrent_jobs": settings.max_concurrent_jobs,
        })

    async def health_handler(request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "service": "clipzy-pipeline",
            "jobs": len({
                r.job_id for r in job_manager._jobs.values()  # noqa: SLF001
                if r.state in (RUNNING,)
            }),
        })

    app.router.add_post("/run", run_handler)
    app.router.add_post("/status", status_handler)
    app.router.add_post("/cancel", cancel_handler)
    app.router.add_post("/probe", probe_handler)
    app.router.add_get("/health", health_handler)
    return app


def serve(settings: WorkerSettings, log_file) -> None:
    """Run the pipeline server standalone (blocking). Writes the readiness
    line the PyWorker LogActionConfig watches for into the model log file."""
    from storage.r2_client import R2Client

    r2 = None
    if settings.r2_configured():
        r2 = R2Client(settings.r2_endpoint_url, settings.r2_access_key_id,
                      settings.r2_secret_access_key, settings.r2_bucket_name,
                      settings.r2_public_domain, settings.r2_region)
        logger.info("Cloudflare R2 client ready (bucket %s).", settings.r2_bucket_name)
    elif settings.r2_required:
        logger.warning("R2 settings incomplete; jobs that need uploads will fail "
                       "until R2_* environment variables are set.")

    job_manager = JobManager(settings, r2=r2)
    app = build_app(settings, job_manager)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write("Clipzy pipeline server starting.\n")

    async def _mark_ready(app: web.Application) -> None:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write("Application startup complete.\n")

    app.on_startup.append(_mark_ready)

    logger.info("Clipzy pipeline server listening on %s:%d (health: /health, jobs: "
                "/run /status /cancel, benchmark: /probe).",
                settings.pipeline_host, settings.pipeline_port)
    web.run_app(app, host=settings.pipeline_host, port=settings.pipeline_port,
                print=None, access_log=None)
