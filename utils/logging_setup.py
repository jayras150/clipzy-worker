"""Structured logging for the Clipzy worker.

Jobs log into their own workspace (job.log) and to stderr; the pipeline server
also mirrors selected lines into the model log file watched by the Vast.ai
PyWorker LogActionConfig (on_load / on_error detection).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"clipzy.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class JobLogger:
    """Logger bound to one job workspace: stderr + work/<id>/job.log."""

    def __init__(self, job_id: str, log_path: Path | None) -> None:
        self.job_id = job_id
        self._logger = get_logger("job")
        self._file_handler: logging.FileHandler | None = None
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
                self._file_handler.setFormatter(logging.Formatter(_FORMAT))
                self._logger.addHandler(self._file_handler)
                self._logger.setLevel(logging.INFO)
            except OSError:
                self._file_handler = None

    def info(self, message: str) -> None:
        self._logger.info("[%s] %s", self.job_id, message)

    def warning(self, message: str) -> None:
        self._logger.warning("[%s] %s", self.job_id, message)

    def error(self, message: str) -> None:
        self._logger.error("[%s] %s", self.job_id, message)

    def debug(self, message: str) -> None:
        self._logger.debug("[%s] %s", self.job_id, message)

    def close(self) -> None:
        if self._file_handler is not None:
            self._logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    warn = warning
