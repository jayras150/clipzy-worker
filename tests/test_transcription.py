"""Transcription engine: merge pipeline, overlap dedupe, cache (mocked API)."""

import json

from engines import transcription
from storage.file_manager import JobWorkspace


class _Settings:
    def __init__(self, cache_dir, tmp_path):
        self.openrouter_api_key = "test-key"
        self.openrouter_api_base = "https://openrouter.example/api/v1"
        self.openrouter_stt_model = "openai/whisper-large-v3-turbo"
        self.local_whisper = False
        self.local_whisper_model = "large-v3-turbo"
        self.local_whisper_device = "cuda"
        self.local_whisper_compute_type = "float16"
        self.cache_dir = cache_dir


def _run_transcribe(tmp_path, monkeypatch, sample_video, offset_chunks=False):
    settings = _Settings(tmp_path / "cache", tmp_path)
    workspace = JobWorkspace(tmp_path / "jobs", "job1").create()
    import shutil
    shutil.copyfile(sample_video, workspace.source)

    source = {"duration": 24.0, "has_audio": True, "source_sha256": "a" * 64}
    chunks = transcription.prepare_audio_chunks(source, workspace)
    assert len(chunks) == 1

    from tests.conftest import canned_transcript_payload

    calls = []

    def fake_remote(chunk, api_base, api_key, stt_model, language_hint, log=print):
        calls.append(chunk)
        # Overlapping duplicate of the first two words (overlap dedupe target)
        payload = canned_transcript_payload(offset=chunk["start"])
        if offset_chunks and chunk["index"] > 0:
            payload["segments"] = payload["segments"][:1]
        return payload

    monkeypatch.setattr(transcription, "_transcribe_chunk_remote", fake_remote)
    transcript = transcription.transcribe_source(source, chunks, settings, workspace, "")
    return transcript, workspace, calls


def test_transcribe_source_merges_segments_and_words(tmp_path, monkeypatch, sample_video):
    transcript, workspace, calls = _run_transcribe(tmp_path, monkeypatch, sample_video)
    assert len(calls) == 1
    assert transcript["language"] == "en"
    assert transcript["word_timings_available"] is True
    assert len(transcript["segments"]) >= 4
    # Words attached to segments with +/- 0.3s tolerance.
    for segment in transcript["segments"]:
        assert "words" in segment
        for word in segment["words"]:
            assert segment["start"] - 0.31 <= word["start"] <= segment["end"] + 0.31
    # transcript.txt written with [mm:ss.ss] prefixes.
    text = workspace.transcript_txt.read_text()
    assert "[00:01.00] Welcome to the channel everyone" in text
    # Cache entry created.
    from cache import cache_manager
    path = cache_manager.CachePaths(tmp_path / "cache", "a" * 64).transcript()
    cached = cache_manager.load_cached(
        path, expect={"source_sha256": "a" * 64, "model": "openai/whisper-large-v3-turbo",
                      "language_hint": "", "pipeline": "v2"})
    assert cached is not None


def test_transcribe_source_cache_hit(tmp_path, monkeypatch, sample_video):
    transcript1, _ws, calls1 = _run_transcribe(tmp_path, monkeypatch, sample_video)
    transcript2, _ws2, calls2 = _run_transcribe(tmp_path, monkeypatch, sample_video)
    # The second run MUST NOT call the API again.
    assert len(calls2) == 0
    assert transcript2["segments"] == transcript1["segments"]


def test_transcribe_refuses_model_substitution(tmp_path, monkeypatch, sample_video):
    settings = _Settings(tmp_path / "cache", tmp_path)
    settings.openrouter_stt_model = "other/model"
    workspace = JobWorkspace(tmp_path / "jobs", "job2").create()
    source = {"duration": 24.0, "has_audio": True, "source_sha256": "b" * 64}
    chunks = [{"index": 0, "file": "x.m4a", "start": 0.0, "duration": 24.0, "overlap": 2.0}]
    import pytest
    with pytest.raises(Exception, match="refusing to transcribe"):
        transcription.transcribe_source(source, chunks, settings, workspace, "")


def test_no_audio_returns_empty_transcript(tmp_path, sample_video):
    settings = _Settings(tmp_path / "cache", tmp_path)
    workspace = JobWorkspace(tmp_path / "jobs", "job3").create()
    transcript = transcription.transcribe_source(
        {"duration": 24.0, "has_audio": False, "source_sha256": "c" * 64},
        [], settings, workspace, "")
    assert transcript["segments"] == []
    assert transcript["word_timings_available"] is False
    assert workspace.transcript_txt.read_text() == ""


def test_transcript_text_sampling():
    segments = [{"start": i * 2.0, "end": i * 2.0 + 1.9, "text": f"segment number {i}"}
                for i in range(2000)]
    text = transcription.build_transcript_text(segments)
    # ~900-segment sample (step = total//900) plus the final segment.
    assert text.count("\n") <= 1002
    assert "transcript sampled for length" in text
    assert text.endswith("segment number 1999")


def test_words_only_regrouped_into_segments():
    # Provider returned only word timings: segments are derived by grouping.
    payload = {"language": "en", "segments": [
        {"start": 0.0, "end": 2.0, "text": "one two three four",
         "words": [{"word": w, "start": 0.1 * i, "end": 0.1 * i + 0.08}
                   for i, w in enumerate(["one", "two", "three", "four"])]}]}
    words = transcription._extract_words(payload, 0.0)
    segments = transcription._extract_segments(payload, 0.0)
    assert len(words) == 4
    assert len(segments) == 1

    # Direct word-list regrouping path (gap > 0.8s or >= 14 words breaks cues).
    merged_words = [
        {"word": "w%d" % i, "start": 0.4 * i, "end": 0.4 * i + 0.3} for i in range(20)]
    current = None
    merged_segments = []
    for word in merged_words:
        if current is None or (word["start"] - current["end"]) > 0.8 or len(current["_words"]) >= 14:
            if current is not None:
                merged_segments.append(current)
            current = {"start": word["start"], "end": word["end"], "text": word["word"],
                       "_words": [word]}
        else:
            current["end"] = max(current["end"], word["end"])
            current["text"] = (current["text"] + " " + word["word"]).strip()
            current["_words"].append(word)
    if current is not None:
        merged_segments.append(current)
    assert len(merged_segments) >= 2
