"""Whisper transcription engine.

Ports the workflow's audio-chunking step (PY_CHUNK) and the OpenRouter
transcription step (PY_TRANSCRIBE): 16 kHz mono AAC chunks of 8 minutes with
a 2 s overlap, verbose_json word+segment timestamps, parallel chunk calls
(MAX_PARALLEL=4), overlap de-duplication with SequenceMatcher, word->segment
regrouping, transcript.txt sampling for the curation prompt, and the
transcript computation cache.

Optional GPU path: when CLIPZY_LOCAL_WHISPER=1 the same merge pipeline is fed
by a local faster-whisper (CUDA) transcription of the same chunks instead of
the OpenRouter API. This is an addition, NOT a substitution: the default
remains openai/whisper-large-v3-turbo through OpenRouter, and the canonical
STT model check still refuses any other configured model name.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import requests

from cache import cache_manager
from config import (CHUNK_SECONDS, EXPECTED_STT_MODEL, MAX_TRANSCRIPT_CHARS,
                    OVERLAP_SECONDS, TRANSCRIPT_CACHE_VER)
from utils import ffmpeg as ff
from utils.logging_setup import get_logger

logger = get_logger("transcription")


class TranscriptionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Audio chunk preparation (PY_CHUNK)
# ---------------------------------------------------------------------------
def prepare_audio_chunks(source: dict, workspace) -> list[dict]:
    ffmpeg = ff.ffmpeg_bin()
    duration = source["duration"]
    has_audio = source["has_audio"]
    audio_dir = Path(workspace.work) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    chunks = []

    if has_audio:
        index = 0
        start = 0.0
        while start < duration - 0.5:
            end = min(duration, start + CHUNK_SECONDS)
            length = end - start
            if length < 1.5:
                break
            path = audio_dir / f"chunk_{index:03d}.m4a"
            cmd = [
                ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(workspace.source),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k", str(path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not path.exists() or path.stat().st_size < 2048:
                raise TranscriptionError(f"failed to extract speech audio chunk {index}: "
                                         f"{proc.stderr[-1500:]}")
            chunks.append({
                "index": index,
                "file": str(path),
                "start": round(start, 3),
                "duration": round(length, 3),
                "overlap": OVERLAP_SECONDS,
            })
            logger.info("Prepared speech audio chunk %d: %.1fs +%.1fs (16 kHz mono AAC).",
                        index, start, length)
            index += 1
            if end >= duration - 0.5:
                break
            start = end - OVERLAP_SECONDS
    else:
        logger.info("Source has no audio stream; transcription will be skipped and no "
                    "subtitles will be burned.")

    with open(workspace.work / "audio_chunks.json", "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, indent=2)
    logger.info("Speech audio plan: %d chunk(s), %.0fs each, %.0fs overlap.",
                len(chunks), CHUNK_SECONDS, OVERLAP_SECONDS)
    return chunks


# ---------------------------------------------------------------------------
# OpenRouter STT (PY_TRANSCRIBE)
# ---------------------------------------------------------------------------
def _transcribe_chunk_remote(chunk, api_base, api_key, stt_model, language_hint, log=print):
    """POST one audio chunk to the OpenRouter transcription endpoint with retries."""
    endpoint = f"{api_base.rstrip('/')}/audio/transcriptions"
    last_error = ""
    for attempt in range(1, 5):
        if attempt > 1:
            delay = min(45, 5 * (2 ** (attempt - 2)))
            log(f"Retry {attempt}/4 for chunk {chunk['index']} in {delay}s (model {stt_model}).")
            time.sleep(delay)
        try:
            with open(chunk["file"], "rb") as fh:
                files = {"file": (os.path.basename(chunk["file"]), fh, "audio/mp4")}
                data = [
                    ("model", stt_model),
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "segment"),
                    ("timestamp_granularities[]", "word"),
                ]
                if language_hint:
                    data.append(("language", language_hint))
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://clipzy.app",
                        "X-Title": "Clipzy Engine",
                    },
                    files=files,
                    data=data,
                    timeout=600,
                )
        except requests.RequestException as exc:
            last_error = f"transport error: {exc}"
            log(f"WARN: chunk {chunk['index']} attempt {attempt}: {last_error}")
            continue
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                last_error = "invalid JSON in 200 response"
                log(f"WARN: chunk {chunk['index']} attempt {attempt}: {last_error}")
                continue
        body = (response.text or "")[:500]
        if response.status_code in (400, 401, 403, 404, 405, 422):
            raise RuntimeError(
                f"Whisper transcription rejected chunk {chunk['index']} "
                f"(HTTP {response.status_code}, model {stt_model}): {body}")
        last_error = f"HTTP {response.status_code}: {body}"
        log(f"WARN: chunk {chunk['index']} attempt {attempt}: {last_error}")
    raise RuntimeError(
        f"Whisper transcription failed for chunk {chunk['index']} after 4 attempts "
        f"using model {stt_model} on {endpoint}. Last error: {last_error}. "
        "The pipeline refuses to switch transcription models; check the OpenRouter "
        "dashboard and retry.")


def _transcribe_chunk_local(chunk, settings):
    """Local faster-whisper transcription of one chunk (GPU when available).

    Produces the same verbose_json-compatible shape the merge logic consumes.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise TranscriptionError(
            "CLIPZY_LOCAL_WHISPER=1 but faster-whisper is not installed "
            "(pip install faster-whisper).") from exc

    device = settings.local_whisper_device if _cuda_available() else "cpu"
    compute = settings.local_whisper_compute_type if device == "cuda" else "int8"
    model = WhisperModel(settings.local_whisper_model, device=device, compute_type=compute)
    segments_iter, info = model.transcribe(
        chunk["file"], word_timestamps=True,
        language=(os.environ.get("SUBTITLE_LANGUAGE") or None))
    payload = {"language": info.language, "segments": []}
    for segment in segments_iter:
        words = []
        for word in (segment.words or []):
            words.append({"word": word.word, "start": word.start, "end": word.end})
        payload["segments"].append({
            "start": segment.start, "end": segment.end,
            "text": segment.text, "words": words,
        })
    return payload


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Merging (verbatim port)
# ---------------------------------------------------------------------------
def _norm_word(text):
    return re.sub(r"[^\w']+", "", str(text or "").lower())


def _extract_words(payload, offset):
    """Collect word-level timings (offset to global time) from verbose_json."""
    words = []

    def absorb(entry):
        try:
            word = str(entry.get("word") or entry.get("text") or "").strip()
            start = float(entry["start"]) + offset
            end = float(entry["end"]) + offset
        except (KeyError, TypeError, ValueError):
            return
        if word and end > start - 0.2:
            words.append({"word": word, "start": round(start, 3), "end": round(end, 3)})

    for segment in payload.get("segments") or []:
        for entry in segment.get("words") or []:
            absorb(entry)
    if not words:
        for entry in payload.get("words") or []:
            absorb(entry)
    return words


def _extract_segments(payload, offset):
    segments = []
    for segment in payload.get("segments") or []:
        try:
            text = str(segment.get("text") or "").strip()
            start = float(segment["start"]) + offset
            end = float(segment["end"]) + offset
        except (KeyError, TypeError, ValueError):
            continue
        if text:
            segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return segments


def _fmt_ts(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def build_transcript_text(segments: list[dict]) -> str:
    lines = [f"[{_fmt_ts(seg['start'])}] {seg['text']}" for seg in segments]
    if lines and sum(len(line) for line in lines) > MAX_TRANSCRIPT_CHARS:
        total = len(lines)
        step = max(1, total // 900)
        sampled = lines[::step]
        if sampled[-1] != lines[-1]:
            sampled.append(lines[-1])
        lines = [f"(transcript sampled for length: showing {len(sampled)} of {total} segments)"] + sampled
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def transcribe_source(source: dict, chunks: list[dict], settings, workspace,
                      language_hint: str) -> dict:
    """Transcribe the source, using (and storing) the computation cache."""
    stt_model = settings.openrouter_stt_model
    cache_path = cache_manager.CachePaths(settings.cache_dir, source["source_sha256"]).transcript()

    cached = cache_manager.load_cached(
        cache_path,
        expect={"source_sha256": source.get("source_sha256"), "model": stt_model,
                "language_hint": language_hint, "pipeline": TRANSCRIPT_CACHE_VER},
        ttl_days=30)
    if cached is not None:
        transcript = cached["transcript"]
        (workspace.work / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        workspace.transcript_txt.write_text(cached.get("transcript_text") or "", encoding="utf-8")
        meta = cached.get("cache_meta") or {}
        logger.info("CACHE HIT: transcript reused for source sha256:%s (model %s); the "
                    "Whisper API was NOT called - %s call(s) and ~%ss saved.",
                    source["source_sha256"][:12], stt_model,
                    meta.get("api_calls", "?"), meta.get("api_seconds", "?"))
        return transcript

    if not chunks:
        transcript = {
            "language": language_hint,
            "word_timings_available": False,
            "chunk_count": 0,
            "segments": [],
            "words": [],
            "note": "no speech audio available",
        }
        (workspace.work / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        workspace.transcript_txt.write_text("", encoding="utf-8")
        logger.info("No speech audio; transcript is empty and subtitles will not be burned.")
        return transcript

    if settings.local_whisper:
        # Local GPU transcription (additive path; same merge logic downstream).
        logger.info("Local faster-whisper transcription enabled (model %s, device auto).",
                    settings.local_whisper_model)
        payloads = []
        for chunk in chunks:
            payloads.append(_transcribe_chunk_local(chunk, settings))
        api_seconds = 0.0
    else:
        if not settings.openrouter_api_key:
            raise TranscriptionError("OPENROUTER_API_KEY is not configured.")
        if stt_model != EXPECTED_STT_MODEL:
            raise TranscriptionError(
                f"refusing to transcribe with '{stt_model}'; Clipzy requires "
                f"'{EXPECTED_STT_MODEL}'. The transcription model must never be "
                "silently substituted.")
        api_started = time.time()
        MAX_PARALLEL = 4

        def _remote(chunk):
            return _transcribe_chunk_remote(
                chunk, settings.openrouter_api_base, settings.openrouter_api_key,
                stt_model, language_hint, log=logger.info)

        try:
            if len(chunks) > 1:
                logger.info("Transcribing %d chunk(s) with %d parallel workers (model %s).",
                            len(chunks), MAX_PARALLEL, stt_model)
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, max(1, len(chunks)))) as pool:
                payloads = list(pool.map(_remote, chunks))
        except RuntimeError as exc:
            raise TranscriptionError(str(exc)) from exc
        api_seconds = time.time() - api_started

    # ---- merge (verbatim port) ------------------------------------------
    merged_words = []
    merged_segments = []
    detected_language = ""
    word_timings_available = False

    for chunk, payload in zip(chunks, payloads):
        served = payload.get("model")
        if served and served != stt_model and not settings.local_whisper:
            logger.warning("upstream served '%s' for a '%s' request.", served, stt_model)
        if not detected_language:
            detected_language = str(payload.get("language") or "").strip()

        offset = chunk["start"]
        overlap = float(chunk.get("overlap") or 0.0)
        chunk_words = _extract_words(payload, offset)
        chunk_segments = _extract_segments(payload, offset)
        if chunk_words:
            word_timings_available = True

        # Remove words duplicated by the chunk overlap before merging.
        if chunk["index"] > 0:
            recent = [(_norm_word(w["word"]), w["start"]) for w in merged_words[-40:]]
            kept_words = []
            for word in chunk_words:
                local_start = word["start"] - offset
                duplicate = local_start < overlap - 0.15
                if not duplicate and local_start < overlap + 0.6:
                    for norm, start in recent:
                        if norm == _norm_word(word["word"]) and abs(start - word["start"]) < 0.75:
                            duplicate = True
                            break
                if not duplicate:
                    kept_words.append(word)
            chunk_words = kept_words

            kept_segments = []
            for segment in chunk_segments:
                local_start = segment["start"] - offset
                duplicate = local_start < overlap - 0.15
                if not duplicate and local_start < overlap + 1.0:
                    for merged in merged_segments[-8:]:
                        if (merged["end"] > segment["start"] - 0.5
                                and SequenceMatcher(None, merged["text"], segment["text"]).ratio() > 0.85):
                            duplicate = True
                            break
                if not duplicate:
                    kept_segments.append(segment)
            chunk_segments = kept_segments

        merged_words.extend(chunk_words)
        merged_segments.extend(chunk_segments)
        logger.info("Transcribed chunk %d: %d segments, %d words.",
                    chunk["index"], len(chunk_segments), len(chunk_words))

    merged_words.sort(key=lambda item: item["start"])

    # Group words into segments when the provider only returned word timings.
    if merged_words and not merged_segments:
        current = None
        for word in merged_words:
            if current is None or (word["start"] - current["end"]) > 0.8 or len(current["_words"]) >= 14:
                if current is not None:
                    merged_segments.append(current)
                current = {"start": word["start"], "end": word["end"], "text": word["word"], "_words": [word]}
            else:
                current["end"] = max(current["end"], word["end"])
                current["text"] = (current["text"] + " " + word["word"]).strip()
                current["_words"].append(word)
        if current is not None:
            merged_segments.append(current)

    merged_segments.sort(key=lambda item: item["start"])

    # Attach word timings to their segments (segment-level timing only otherwise).
    for segment in merged_segments:
        seg_words = [w for w in merged_words
                     if segment["start"] - 0.3 <= w["start"] <= segment["end"] + 0.3]
        segment["words"] = seg_words
        segment.pop("_words", None)

    resolved_language = language_hint or detected_language or ""

    transcript = {
        "language": resolved_language,
        "detected_language": detected_language,
        "word_timings_available": word_timings_available,
        "chunk_count": len(chunks),
        "model": stt_model,
        "segments": merged_segments,
        "words": merged_words,
    }

    (workspace.work / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    transcript_text = build_transcript_text(merged_segments)
    workspace.transcript_txt.write_text(transcript_text, encoding="utf-8")

    cache_manager.save_cached(
        cache_path,
        {"transcript": transcript, "transcript_text": transcript_text},
        meta={"source_sha256": source.get("source_sha256"), "model": stt_model,
              "language_hint": language_hint, "pipeline": TRANSCRIPT_CACHE_VER,
              "api_calls": len(chunks), "api_seconds": round(api_seconds, 1)})

    logger.info("Transcript complete: %d segments, %d words, language=%s, "
                "word_timings=%s (%d Whisper call(s) in %.1fs; cached for this source hash).",
                len(merged_segments), len(merged_words), resolved_language or "unknown",
                word_timings_available, len(chunks), api_seconds)
    return transcript
