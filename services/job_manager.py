"""Async job manager with isolated execution and Supabase lifecycle callbacks."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config import WorkerSettings
from pipeline.runner import JobCancelled, JobFailure, run_pipeline
from schemas.job_request import JobRequest
from services.progress import ProgressTracker
from services.supabase_callback import SupabaseCallback
from storage.file_manager import JobWorkspace
from utils.logging_setup import JobLogger, get_logger

logger = get_logger("job_manager")

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"


class JobRecord:
    def __init__(self, job_id: str, request: JobRequest) -> None:
        self.job_id = job_id
        self.request = request
        self.state = PENDING
        self.progress = 0
        self.stage = ""
        self.stage_label = ""
        self.response: dict | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.cancel_event = threading.Event()

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.state,
            "progress": self.progress,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "response": self.response,
            "error": self.error,
        }


class JobManager:
    def __init__(self, settings: WorkerSettings, r2=None) -> None:
        self.settings = settings
        self.r2 = r2
        self.callback = SupabaseCallback(settings)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs),
            thread_name_prefix="clipzy-job",
        )
        self._log = get_logger("job_manager")

    def _notify(self, record: JobRecord, event_type: str, message: str = "") -> None:
        self.callback.send(
            record.job_id,
            event_type,
            status=record.state,
            progress=record.progress,
            current_stage=record.stage,
            stage_label=record.stage_label,
            message=message,
            error_message=record.error,
            response=record.response if event_type == "completed" else None,
        )

    def submit(self, request: JobRequest) -> JobRecord:
        job_id = request.job_id
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing and existing.state in (RUNNING, PENDING):
                raise RuntimeError(f"job '{job_id}' is already running on this worker")
            record = JobRecord(job_id, request)
            self._jobs[job_id] = record
        self._executor.submit(self._run, record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None:
            return False
        if record.state in (RUNNING, PENDING):
            record.cancel_event.set()
            return True
        return False

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._jobs.values() if r.state in (RUNNING, PENDING))

    def prune(self) -> None:
        cutoff = time.time() - self.settings.job_keep_seconds
        with self._lock:
            for job_id in list(self._jobs):
                record = self._jobs[job_id]
                if record.state in (COMPLETED, FAILED, CANCELLED) and (record.finished_at or 0) < cutoff:
                    del self._jobs[job_id]

    def _run(self, record: JobRecord) -> None:
        job_id = record.job_id
        log_path = JobWorkspace(self.settings.work_root, job_id).root / "job.log"
        job_logger = JobLogger(job_id, log_path)
        record.state = RUNNING
        record.started_at = time.time()
        self._log.info("job %s started (worker thread)", job_id)
        self._notify(record, "started", "Worker started processing the job.")

        def on_update(stage_id: str, stage_label: str, percent: int) -> None:
            record.stage = stage_id
            record.stage_label = stage_label
            record.progress = percent
            self._push_intermediate_status(record)
            self._notify(record, "progress")

        progress = ProgressTracker(job_id, on_update=on_update)
        record.stage, record.stage_label = "validate", "Validating job parameters"
        record.progress = 2
        self._notify(record, "progress")

        try:
            response = run_pipeline(
                record.request,
                self.settings,
                progress,
                job_logger,
                cancel_check=record.cancel_event.is_set,
                r2=self.r2,
            )
            record.response = response
            record.state = COMPLETED
            record.progress = 100
            record.stage = "completed"
            record.stage_label = "Processing completed"
            record.finished_at = time.time()
            self._log.info(
                "job %s completed in %.1fs",
                job_id,
                record.finished_at - record.started_at,
            )
            self._notify(record, "completed", "Video processing completed.")
        except JobCancelled as exc:
            record.state = CANCELLED
            record.error = str(exc)[:2000]
            record.finished_at = time.time()
            self._log.info("job %s cancelled", job_id)
            self._notify(record, "cancelled", record.error)
        except Exception as exc:  # noqa: BLE001
            record.state = FAILED
            record.error = str(exc)[:2000]
            record.finished_at = time.time()
            self._log.error("job %s failed: %s", job_id, record.error)
            self._notify(record, "failed", record.error)
        finally:
            job_logger.close()

    def _push_intermediate_status(self, record: JobRecord) -> None:
        if not self.settings.status_uploads or self.r2 is None:
            return
        from pipeline.runner import upload_intermediate_status

        upload_intermediate_status(
            record.request,
            self.settings,
            record.progress,
            record.stage_label,
            self.r2,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
