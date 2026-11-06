"""Clipzy GPU Processing Worker - Vast.ai Serverless PyWorker entrypoint.

Implements the official Vast.ai PyWorker interface (vastai SDK):
WorkerConfig / HandlerConfig / BenchmarkConfig / LogActionConfig, deployed by
setting PYWORKER_REPO to the repository containing this file (see
docs.vast.ai/guides/serverless/worker-py).

Architecture on each worker instance:

  +---------------------------------------------------------------+
  | Vast serverless router -> PyWorker (this file, WORKER_PORT)   |
  |     routes: /run /status /cancel /probe                       |
  |     + sessions (/session/*), auth, queueing, workload         |
  +-------------------------------+-------------------------------+
                                  | forwards payload
                                  v
  | Clipzy pipeline server (internal "model server", 127.0.0.1:8080)|
  |     full processing pipeline, job manager, R2 uploads          |

Long jobs (10-90 minutes) use the documented asynchronous-task session
pattern: the client opens a session, POSTs /run once (returns immediately),
polls /status, and config on_close_route="/cancel" guarantees cleanup when
the client disappears (docs.vast.ai/guides/serverless/sessions).

Standalone mode (no Vast, e.g. local Docker or bare metal):
    python worker.py --standalone
runs the pipeline server directly on 0.0.0.0:8080.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time

from config import WorkerSettings
from utils.logging_setup import get_logger

logger = get_logger("worker")

PIPELINE_LOAD_LOG_MSGS = [
    "Application startup complete.",
]

PIPELINE_ERROR_LOG_MSGS = [
    "Traceback (most recent call last):",
    "Clipzy pipeline server failed",
    "ERROR: quality control failed",
]

PIPELINE_INFO_LOG_MSGS = [
    "progress",
    "CACHE HIT",
]


def _start_pipeline_server(settings: WorkerSettings) -> threading.Thread:
    """Run the internal pipeline ("model") server in a background thread."""
    from aiohttp import web

    from services.job_manager import JobManager
    from services.pipeline_server import build_app
    from storage.r2_client import R2Client

    r2 = None
    if settings.r2_configured():
        r2 = R2Client(settings.r2_endpoint_url, settings.r2_access_key_id,
                      settings.r2_secret_access_key, settings.r2_bucket_name,
                      settings.r2_public_domain, settings.r2_region)
    job_manager = JobManager(settings, r2=r2)
    app = build_app(settings, job_manager)

    log_file = settings.pipeline_log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    async def _main() -> None:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write("Clipzy pipeline server starting.\n")
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, settings.pipeline_host, settings.pipeline_port)
        await site.start()
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write("Application startup complete.\n")
        logger.info("Clipzy pipeline server listening on %s:%d.",
                    settings.pipeline_host, settings.pipeline_port)
        await asyncio.Event().wait()

    def _thread_target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_main())
        except Exception as exc:  # noqa: BLE001
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(f"Clipzy pipeline server failed: {exc}\n")
            logger.error("pipeline server thread died: %s", exc)

    thread = threading.Thread(target=_thread_target, name="clipzy-pipeline", daemon=True)
    thread.start()
    return thread


def _wait_for_pipeline(settings: WorkerSettings, timeout: float = 120.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    url = f"http://{settings.pipeline_host}:{settings.pipeline_port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def run_vast_pyworker() -> None:
    """Build the WorkerConfig and hand control to the vastai SDK."""
    try:
        from vastai import (BenchmarkConfig, HandlerConfig, LogActionConfig,
                            Worker, WorkerConfig)
    except ImportError as exc:
        raise SystemExit(
            "The 'vastai' SDK package is required to run the Vast.ai PyWorker "
            "(pip install vastai-sdk). Use `python worker.py --standalone` for "
            "non-Vast deployments.") from exc

    settings = WorkerSettings()
    settings.ensure_dirs()

    # ---- resource provisioning before readiness ------------------------
    from services import resources

    resources.ensure_fonts(settings.fonts_dir)
    from engines import face_detection as fd

    fd.ensure_yunet_model(settings.models_dir)

    # ---- internal "model server" (the full Clipzy pipeline) ------------
    _start_pipeline_server(settings)
    if not _wait_for_pipeline(settings):
        raise SystemExit("internal pipeline server did not become healthy in time")

    # ---- workload: constant per request (job runtime is dominated by a  ---
    # single long background pipeline run, not per-request model compute).
    def workload(payload: dict) -> float:
        return 100.0

    worker_config = WorkerConfig(
        model_server_url=f"http://{settings.pipeline_host}",
        model_server_port=settings.pipeline_port,
        model_log_file=str(settings.pipeline_log_file),
        model_healthcheck_url="/health",
        handlers=[
            # /probe is the benchmark handler: a lightweight roundtrip that
            # reports GPU state without starting a real job.
            HandlerConfig(
                route="/probe",
                allow_parallel_requests=True,
                max_queue_time=60.0,
                workload_calculator=workload,
                benchmark_config=BenchmarkConfig(
                    dataset=[{"probe": True}],
                    runs=1,
                ),
            ),
            HandlerConfig(
                route="/run",
                allow_parallel_requests=True,
                max_queue_time=60.0,
                workload_calculator=workload,
            ),
            HandlerConfig(
                route="/status",
                allow_parallel_requests=True,
                max_queue_time=30.0,
                workload_calculator=lambda payload: 1.0,
            ),
            HandlerConfig(
                route="/cancel",
                allow_parallel_requests=True,
                max_queue_time=30.0,
                workload_calculator=lambda payload: 1.0,
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=PIPELINE_LOAD_LOG_MSGS,
            on_error=PIPELINE_ERROR_LOG_MSGS,
            on_info=PIPELINE_INFO_LOG_MSGS,
        ),
    )
    logger.info("Starting Vast.ai PyWorker (routes /run /status /cancel /probe; "
                "pipeline backend 127.0.0.1:%d).", settings.pipeline_port)
    Worker(worker_config).run()


def run_standalone(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Run the pipeline server directly (local Docker / bare metal)."""
    from services.pipeline_server import serve

    settings = WorkerSettings()
    settings.ensure_dirs()
    if port is not None:
        settings.pipeline_port = port
    if host:
        settings.pipeline_host = host

    from services import resources

    resources.ensure_fonts(settings.fonts_dir)
    from engines import face_detection as fd

    fd.ensure_yunet_model(settings.models_dir)
    serve(settings, settings.pipeline_log_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clipzy GPU processing worker")
    parser.add_argument("--standalone", action="store_true",
                        help="run the pipeline HTTP server directly (no Vast.ai SDK)")
    parser.add_argument("--host", default="0.0.0.0", help="standalone bind host")
    parser.add_argument("--port", type=int, default=None, help="standalone bind port")
    args = parser.parse_args()
    if args.standalone:
        run_standalone(host=args.host, port=args.port)
    else:
        run_vast_pyworker()


if __name__ == "__main__":
    main()
