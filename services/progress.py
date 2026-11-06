"""Job progress tracking.

Stage-based progress model spanning the whole pipeline; every transition is
written into the in-memory job record (served by /status) and, when enabled,
pushed as an intermediate status.json to R2 so the Android app can poll job
progress exactly like it polled the final status object before.
"""

from __future__ import annotations

from utils.logging_setup import get_logger

logger = get_logger("progress")

# (stage_id, label, percent-at-completion)
STAGES = [
    ("validate", "Validating job parameters", 2),
    ("download", "Downloading source video", 12),
    ("probe", "Probing source and fingerprinting", 16),
    ("audio", "Preparing speech audio chunks", 20),
    ("transcribe", "Transcribing speech with Whisper", 34),
    ("vision", "Extracting keyframes and scene cuts", 42),
    ("faces", "Scanning for faces", 47),
    ("curate", "Curating viral clips with GPT-6 Luna", 58),
    ("track", "Tracking faces inside selected clips", 63),
    ("zoom", "Reconciling zoom decisions and geometry", 66),
    ("subs", "Generating styled subtitles", 70),
    ("refine", "Refining clip metadata", 74),
    ("plan", "Building render plan", 77),
    ("render", "Rendering clips", 92),
    ("qc", "Quality control", 96),
    ("manifest", "Building Android manifest", 98),
    ("upload", "Uploading clips and metadata to R2", 100),
]
_STAGE_MAP = {stage[0]: stage for stage in STAGES}


class ProgressTracker:
    def __init__(self, job_id: str, on_update=None) -> None:
        self.job_id = job_id
        self._on_update = on_update
        self._stage_id = "validate"
        self._percent = 0

    def stage(self, stage_id: str) -> None:
        entry = _STAGE_MAP.get(stage_id)
        if entry is None:
            return
        self._stage_id, _, self._percent = entry
        logger.info("[%s] progress %d%% - %s", self.job_id, self._percent, entry[1])
        if self._on_update:
            try:
                self._on_update(self._stage_id, entry[1], self._percent)
            except Exception:  # noqa: BLE001 - progress must never kill a job
                pass

    def snapshot(self) -> dict:
        entry = _STAGE_MAP.get(self._stage_id, ("", "", self._percent))
        return {"stage": self._stage_id, "stage_label": entry[1],
                "progress": self._percent}
