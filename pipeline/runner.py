"""The complete Clipzy processing pipeline (sequential orchestrator).

Maps 1:1 to the workflow steps, in the same order, with the same artifacts
written under the job workspace (work/params.json, work/source.json,
work/transcript.json, transcript.txt, work/frames.json, work/face_context.json,
work/face_tracks.json, work/curation.json, work/video_analysis.json,
work/render_hints.json, work/clips_final.json, work/subtitles.json,
work/render_plan.json, work/outputs_probe.json, output_clips/clip_N.mp4,
clips_manifest.json, status.json).

GitHub-Actions-specific mechanics (GITHUB_ENV hand-off, actions/cache
restore/save, upload-artifact) are replaced by their native equivalents:
an in-process context object, the persistent cache manager, and R2 debug
uploads.
"""

from __future__ import annotations

import json
import time

from config import WorkerSettings
from engines import face_detection, transcription, viral_detection, vision_engine
from processors import (auto_crop, auto_zoom, downloader, renderer, source_probe,
                        subtitle_generator, watermark)
from schemas import job_response
from schemas.job_request import JobRequest
from services import manifest as manifest_builder
from services import resources
from services.progress import ProgressTracker
from storage.file_manager import JobWorkspace, dir_size
from utils.logging_setup import JobLogger, get_logger

logger = get_logger("pipeline")


class JobCancelled(Exception):
    pass


class JobFailure(Exception):
    pass


class PipelineContext:
    """Everything the workflow passed through GITHUB_ENV/files, as one object."""

    def __init__(self, request: JobRequest, settings: WorkerSettings, workspace: JobWorkspace):
        self.request = request
        self.params = request.params
        self.settings = settings
        self.workspace = workspace
        self.job_id = request.job_id
        self.source: dict = {}
        self.transcript: dict = {}
        self.chunks: list = []
        self.frames_manifest: dict = {}
        self.face_context: dict = {}
        self.clips_kept: list = []
        self.video_analysis: dict = {}
        self.tracks: list = []
        self.geometry: dict = {}
        self.clips_final: list = []
        self.render_hints: dict = {}
        self.subtitles: dict = {}
        self.plan: dict = {}
        self.probes: list = []
        self.manifest: list = []
        self.watermark_file = None
        self.started = time.time()

    # -- artifacts the workflow wrote for later steps ---------------------
    def persist(self, name: str, document) -> None:
        path = self.workspace.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    def log_upload_summary(self, urls: dict) -> None:
        logger.info("Upload complete for job: %s (primary: %s, manifest: %s)",
                    self.job_id, urls.get("primary"), urls.get("manifest"))


def run_pipeline(request: JobRequest, settings: WorkerSettings,
                 progress: ProgressTracker, log: JobLogger,
                 cancel_check=None, r2=None) -> dict:
    """Execute the full pipeline for one job. Returns the job response dict.

    Raises JobCancelled when cancel_check() turns true between stages, and
    JobFailure on any pipeline error (after uploading debug artifacts).
    """
    def check_cancel():
        if cancel_check and cancel_check():
            raise JobCancelled(f"job {request.job_id} cancelled by request")

    workspace = JobWorkspace(settings.work_root, request.job_id).create()
    log.info("Job accepted. Normalized parameters:")
    for key, value in request.summary().items():
        log.info(f"  {key} = {value}")
    if request.params.get("manual_transform"):
        mt = request.params["manual_transform"]
        log.info(f"Manual transform accepted: scale={mt['scale']}, offsets="
                 f"({mt['offset_x']}, {mt['offset_y']}) (applied as the base framing for "
                 f"every rendered clip).")
    else:
        log.info("Manual transform: none (default framing).")
    ctx = PipelineContext(request, settings, workspace)
    ctx.persist("params.json", request.params)

    failed_error = None
    try:
        # ---- download ---------------------------------------------------
        progress.stage("download")
        check_cancel()
        downloader.download_source(
            request.youtube_url, request.params.get("output_quality", "original"),
            settings.yt_cookies, workspace.root)
        # The verified file must live at the canonical source path.
        source_file = workspace.root / "raw_video.mp4"
        if not source_file.exists():
            raise JobFailure("verified source file missing after download")

        # ---- probe --------------------------------------------------------
        progress.stage("probe")
        check_cancel()
        ctx.source = source_probe.probe_source(source_file)
        ctx.persist("source.json", ctx.source)

        # ---- audio chunks + watermark -------------------------------------
        progress.stage("audio")
        check_cancel()
        ctx.chunks = transcription.prepare_audio_chunks(ctx.source, workspace)
        ctx.watermark_file = watermark.prepare_watermark(request.params, workspace.work)

        # ---- transcription --------------------------------------------------
        progress.stage("transcribe")
        check_cancel()
        ctx.transcript = transcription.transcribe_source(
            ctx.source, ctx.chunks, settings, workspace,
            request.params.get("subtitle_language", ""))
        check_cancel()

        # ---- vision (keyframes + scene cuts) --------------------------------
        progress.stage("vision")
        ctx.frames_manifest = vision_engine.extract_keyframes(ctx.source, workspace, settings)
        check_cancel()

        # ---- coarse face scan -------------------------------------------------
        progress.stage("faces")
        engine = face_detection.FaceEngine(settings.models_dir)
        ctx.face_context = face_detection.coarse_face_scan(
            ctx.source, ctx.frames_manifest, engine, settings)
        check_cancel()

        # ---- curation -----------------------------------------------------------
        progress.stage("curate")
        transcript_text = workspace.transcript_txt.read_text(encoding="utf-8").strip()
        ctx.clips_kept, ctx.video_analysis = viral_detection.curate_clips(
            ctx.source, ctx.frames_manifest, ctx.face_context, ctx.transcript,
            transcript_text, request.params, settings, workspace)
        check_cancel()

        # ---- in-clip face tracking -----------------------------------------------
        progress.stage("track")
        ctx.tracks = face_detection.track_faces_in_clips({"clips": ctx.clips_kept}, engine, workspace)
        check_cancel()

        # ---- zoom + geometry reconciliation ----------------------------------------
        progress.stage("zoom")
        tracks_map = {t["clip_index"]: t for t in ctx.tracks}
        ctx.geometry = auto_crop.compute_geometry(ctx.source, request.params, ctx.face_context)
        ctx.clips_final, ctx.render_hints = auto_zoom.reconcile_zooms(
            ctx.source, {"clips": ctx.clips_kept}, tracks_map, ctx.face_context,
            request.params, ctx.geometry)
        ctx.persist("render_hints.json", ctx.render_hints)
        ctx.persist("clips_final.json", ctx.clips_final)
        check_cancel()

        # ---- fonts + subtitles --------------------------------------------------------
        progress.stage("subs")
        resources.ensure_fonts(settings.fonts_dir)
        ctx.subtitles = subtitle_generator.generate_subtitles(
            request.params, ctx.render_hints, ctx.clips_final, ctx.transcript,
            workspace.work / "subs")
        ctx.persist("subtitles.json", ctx.subtitles)
        check_cancel()

        # ---- metadata refinement ---------------------------------------------------------
        progress.stage("refine")
        ctx.clips_final = viral_detection.refine_metadata(
            ctx.clips_final, ctx.transcript, tracks_map, ctx.video_analysis, settings, workspace)
        ctx.persist("clips_final.json", ctx.clips_final)
        check_cancel()

        # ---- render plan + render ------------------------------------------------------------
        progress.stage("plan")
        ctx.plan = renderer.build_render_plan(
            ctx.source, ctx.render_hints, ctx.clips_final, ctx.subtitles,
            request.params, workspace, settings, ctx.watermark_file)
        check_cancel()

        progress.stage("render")
        ctx.probes = renderer.render_clips(ctx.plan, workspace)
        check_cancel()

        # ---- cache usage report (workflow parity) --------------------------------------------
        _cache_report(settings, log)

        # ---- quality control ----------------------------------------------------------------------
        progress.stage("qc")
        check_cancel()
        probes_map = {p["clip_index"]: p for p in ctx.probes}
        errors, warnings = _run_qc(ctx, probes_map)
        if errors:
            for error in errors:
                log.error(f"QC: {error}")
            raise JobFailure("quality control failed: " + "; ".join(errors[:10]))

        # ---- manifest ---------------------------------------------------------------------------------
        progress.stage("manifest")
        ctx.manifest = manifest_builder.build_manifest(
            ctx.clips_final, ctx.subtitles, probes_map, ctx.transcript,
            request.params, tracks_map, workspace)
        check_cancel()

        # ---- upload ----------------------------------------------------------------------------------
        progress.stage("upload")
        if settings.r2_required:
            if r2 is None:
                raise JobFailure("R2 storage is required but not configured.")
            _upload_outputs(ctx, r2, progress)
        else:
            log.warning("CLIPZY_R2_REQUIRED=false: skipping uploads; URLs will be empty.")

        processing_time = time.time() - ctx.started
        response = job_response.build_job_response(
            request.job_id, "completed", ctx.manifest, ctx.manifest,
            settings.r2_public_domain if settings.r2_required else "",
            processing_time, progress=100,
            transcript_language=ctx.transcript.get("language", ""), params=request.params)
        log.info(f"Job completed in {processing_time:.1f}s: {len(ctx.manifest)} clip(s).")
        return response

    except JobCancelled as exc:
        failed_error = f"cancelled: {exc}"
        raise
    except Exception as exc:  # noqa: BLE001
        failed_error = str(exc)
        raise
    finally:
        # Debug artifacts (replaces upload-artifact on failure): the raw GPT-6
        # Luna attempts plus the render plan/hints/manifest. Never secrets.
        try:
            if failed_error is not None and settings.r2_required and r2 is not None:
                r2.upload_debug(request.job_id, workspace.work / "debug", [
                    workspace.work / "render_plan.json",
                    workspace.work / "render_hints.json",
                    workspace.manifest,
                ])
        except Exception as exc:  # noqa: BLE001
            log.warning("debug artifact upload failed: %s", exc)
        try:
            if failed_error is not None and settings.r2_required and r2 is not None:
                r2.upload_status(request.job_id, job_response.build_status_document(
                    request.job_id, "failed", settings.r2_public_domain, [], request.params,
                    progress=progress.snapshot()["progress"], error=failed_error))
        except Exception as exc:  # noqa: BLE001
            log.warning("failure status upload failed: %s", exc)
        workspace.cleanup()
        log.close()


def _run_qc(ctx: PipelineContext, probes_map: dict) -> tuple[list, list]:
    from services import quality_control

    return quality_control.run_quality_control(
        ctx.plan, probes_map, ctx.subtitles, ctx.clips_final, ctx.params, ctx.workspace)


def _upload_outputs(ctx: PipelineContext, r2, progress: ProgressTracker) -> None:
    settings = ctx.settings
    request = ctx.request
    urls = r2.upload_job_outputs(request.job_id, ctx.workspace.output_clips, ctx.manifest)
    ctx.log_upload_summary(urls)
    if settings.status_uploads:
        status_doc = job_response.build_status_document(
            request.job_id, "ready", settings.r2_public_domain, ctx.manifest,
            request.params, transcript_language=ctx.transcript.get("language", ""),
            progress=100)
        r2.upload_status(request.job_id, status_doc)


def _cache_report(settings: WorkerSettings, log) -> None:
    """Report cache usage (videos are never cached - only reusable resources
    and computation results keyed by source content hash)."""
    from pathlib import Path

    entries = (
        ("computation: transcript", Path(settings.cache_dir) / "transcript"),
        ("computation: curation", Path(settings.cache_dir) / "curation"),
        ("computation: vision", Path(settings.cache_dir) / "vision"),
        ("resource: YuNet model", Path(settings.models_dir)),
        ("resource: fonts", Path(settings.fonts_dir)),
    )
    log.info("CACHE REPORT (videos are never cached - only reusable resources and "
             "computation results keyed by source content hash):")
    for label, path in entries:
        size, count = dir_size(path)
        log.info(f"  {label:32s} {count:4d} file(s)  {size / 1e6:8.2f} MB")


def upload_intermediate_status(request: JobRequest, settings: WorkerSettings,
                               percent: int, stage_label: str, r2) -> None:
    """Progress callback body: push an intermediate processing status.json."""
    if r2 is None or not settings.status_uploads:
        return
    try:
        r2.upload_status(request.job_id, job_response.build_status_document(
            request.job_id, "processing", settings.r2_public_domain, [], request.params,
            progress=percent, stage=stage_label))
    except Exception as exc:  # noqa: BLE001 - progress uploads never kill a job
        logger.warning("intermediate status upload failed: %s", exc)
