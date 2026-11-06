"""Viral clip curation with GPT-6 Luna (multimodal) + metadata refinement.

Verbatim port of PY_CURATE and PY_META, including the full prompt text, the
three-attempt retry ladder (full -> continue-from-salvage -> compact), the
finish_reason==length truncated-JSON salvage, hook verbatim verification,
zoom-intent normalization, overlap de-duplication and the curation
computation cache.
"""

from __future__ import annotations

import base64
import json
import re
import time
from difflib import SequenceMatcher

from cache import cache_manager
from config import (CURATION_PROMPT_VERSION, CURATION_TTL_DAYS, EXPECTED_CHAT_MODEL,
                    MAX_CURATION_TOKENS, MAX_REFINE_TOKENS)
from engines import gpt_client
from utils.logging_setup import get_logger

logger = get_logger("curation")


class CurationError(RuntimeError):
    pass


def _parse_target_duration(target_duration: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", target_duration or "30-60")
    low = int(match.group(1)) if match else 30
    high = int(match.group(2) or low) if match else low
    return low, high


def _hook_is_real(hook, clip_start, clip_end, segments):
    if not hook:
        return False
    window = " ".join(s["text"] for s in segments
                      if s["end"] > clip_start - 8 and s["start"] < clip_end + 8).lower()
    probe = re.sub(r"\s+", " ", hook.lower()).strip(" .!?,")
    if not probe:
        return False
    size = len(probe)
    for start in range(0, max(1, len(window) - size + 1)):
        if SequenceMatcher(None, probe, window[start:start + size]).ratio() > 0.8:
            return True
    return False


def _sanitize_tags(tags) -> list[str]:
    clean = []
    if isinstance(tags, list):
        for tag in tags[:5]:
            tag = str(tag).strip().replace(" ", "")
            if not tag:
                continue
            if not tag.startswith("#"):
                tag = "#" + tag.lstrip("#")
            if re.fullmatch(r"#[\w\u00c0-\uffff]+", tag, re.IGNORECASE) and tag not in clean:
                clean.append(tag)
    return clean


# ---------------------------------------------------------------------------
# Curation (PY_CURATE)
# ---------------------------------------------------------------------------
def curate_clips(source: dict, frames_manifest: dict, face_context: dict,
                 transcript: dict, transcript_text: str, params: dict,
                 settings, workspace) -> tuple[list, dict]:
    """Returns (clips, video_analysis). Raises CurationError when the model
    produces nothing usable - fallback clips are NEVER fabricated."""
    chat_model = settings.openrouter_chat_model
    if not settings.openrouter_api_key:
        raise CurationError("OPENROUTER_API_KEY is not configured.")
    if chat_model != EXPECTED_CHAT_MODEL:
        raise CurationError(
            f"refusing to run curation with '{chat_model}'; Clipzy requires "
            f"'{EXPECTED_CHAT_MODEL}'. The chat/vision model must never be silently "
            "substituted.")

    target_duration = params.get("target_duration", "30-60")
    duration = source["duration"]
    language = transcript.get("language") or "unknown"
    language_hint = params.get("subtitle_language", "")
    low, high = _parse_target_duration(target_duration)

    cache_paths = cache_manager.CachePaths(settings.cache_dir, source["source_sha256"])
    cached = cache_manager.load_cached(
        cache_paths.curation(),
        expect={"source_sha256": source.get("source_sha256"), "model": chat_model,
                "target_duration": target_duration, "language_hint": language_hint,
                "prompt_version": CURATION_PROMPT_VERSION},
        ttl_days=CURATION_TTL_DAYS)
    if cached is not None:
        (workspace.work / "curation.json").write_text(
            json.dumps({"clips": cached["clips"]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (workspace.work / "video_analysis.json").write_text(
            json.dumps(cached.get("video_analysis") or {}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        meta = cached.get("cache_meta") or {}
        logger.info("CACHE HIT: curation reused for source sha256:%s (model %s, target %s); "
                    "the GPT-6 Luna API was NOT called - ~%ss saved.",
                    source["source_sha256"][:12], chat_model, target_duration,
                    meta.get("api_seconds", "?"))
        return cached["clips"], cached.get("video_analysis") or {}

    if face_context["frames_with_faces_pct"] < 5:
        face_summary = "no faces detected in sampled frames"
    else:
        face_summary = (f"faces visible in {face_context['frames_with_faces_pct']:.0f}% of "
                        f"sampled frames (avg {face_context['avg_faces_per_frame']:.1f} per "
                        f"frame, detector {face_context['method']})")

    frame_lines = [f"{i + 1}. t={frame['t']}s" for i, frame in enumerate(frames_manifest["ai_frames"])]
    frames_block = "\n".join(frame_lines) or "(no frames could be extracted)"

    prompt = f"""You are Clipzy's senior short-form video editor. Analyze this video using BOTH the attached keyframes (vision) and the timestamped transcript (audio), then select genuinely viral-worthy clips.

VIDEO METADATA
- duration: {duration:.1f} seconds
- resolution: {source['width']}x{source['height']} ({source['aspect_ratio']})
- frame rate: {source['fps']:g} fps
- spoken language: {language} (MANDATORY: ALL TITLES, DESCRIPTIONS, TOPICS, AND HASHTAGS MUST BE WRITTEN IN THIS EXACT SPOKEN LANGUAGE, E.G. BAHASA INDONESIA IF SPOKEN LANGUAGE IS INDONESIAN)
- faces: {face_summary}

KEYFRAMES (attached as images, in timeline order)
{frames_block}

TRANSCRIPT (timestamped [mm:ss.ss], possibly truncated or empty)
{transcript_text or '(no speech detected)'}

TASK 1 - VIDEO UNDERSTANDING: describe the topic, a short summary, the main subjects (use role descriptions like "the host" or "the male guest" unless a name is spoken or visibly shown - never guess real-world identity), the scenes, and notable visual events including on-screen text.

TASK 2 - CLIP SELECTION: select only genuinely interesting moments: strong hooks, surprising statements, educational explanations, funny moments, emotional reactions, important revelations, controversial discussions, unexpected events, interesting opinions, storytelling peaks, strong conclusions.

CLIP RULES:
- Return 1 to 8 clips. Quality beats quantity: if only three moments deserve clipping, return three. Never manufacture filler clips.
- Every clip must be a complete thought: start at a natural sentence/beat start and end after the thought completes (align to transcript timestamps). Never cut mid-sentence.
- start_sec >= 0, end_sec <= {duration:.1f}, duration between {low} and {high} seconds (use the longest valid duration if the source is shorter).
- No duplicate or heavily overlapping clips, no long silences.
- viral_score: honest integer 1-100.
- topic: the specific subject of THIS clip.
- hook: a short quote VERBATIM from the transcript inside the clip, or "" if unsure. Never invent quotes.
- title: HIGH-CTR & VIRAL. Specific, engaging, curiosity-inducing, max 60 characters, written in the EXACT SPOKEN LANGUAGE of the video (e.g., Bahasa Indonesia for Indonesian videos). Never use generic clickbait like "Amazing Video" or "Video Highlight". No generic English titles for non-English videos.
- description: accurate, engaging summary of what is actually said/shown in the clip in the video's spoken language, max 200 characters.
- tags: 4-5 niche-specific hashtags strictly grounded in the clip's topic, category and keywords (e.g. #investasi #keuangan #saham #bisnis for finance videos in Indonesian). NEVER use generic placeholders like #shorts #viral #fyp unless strictly relevant.
- visual_analysis: main_subject and scene grounded in the keyframes nearest to the clip.

ZOOM RULES (CRITICAL):
- zoom_intents: ONLY for moments where a meaningful VISUAL event justifies emphasis (visible reaction close-up, important object revealed on screen, dramatic on-screen moment, key speaker emphasis).
- If there is no clear visual justification, return an empty zoom_intents array. No zoom is the correct default; never guess zoom moments to fill space.
- Each intent: start_rel_sec and end_rel_sec relative to the clip start (0 <= start < end <= clip duration), zoom_level from {{1.10, 1.15, 1.20}} (prefer 1.10 and 1.15; use 1.20 sparingly), duration 2-15 seconds, at most 3 intents per clip, target "primary_subject", and a concrete reason grounded in the frames/transcript.

OUTPUT: a single JSON object exactly shaped like:
{{"video_analysis": {{"topic": "", "summary": "", "main_subjects": [], "scenes": [], "language": ""}}, "clips": [{{"clip_index": 1, "start_sec": 0.0, "end_sec": 0.0, "viral_score": 0, "topic": "", "hook": "", "title": "", "description": "", "tags": ["#topic"], "visual_analysis": {{"main_subject": "", "scene": ""}}, "zoom_intents": [{{"start_rel_sec": 0.0, "end_rel_sec": 0.0, "zoom_level": 1.15, "target": "primary_subject", "reason": ""}}]}}]}}
Return only JSON. No Markdown fences, no commentary."""

    parts = [{"type": "text", "text": prompt}]
    for frame in frames_manifest["ai_frames"]:
        with open(frame["file"], "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

    system_message = ("You are Clipzy's senior short-form video editor and virality strategist. "
                      "You return ONLY valid JSON, never Markdown fences or commentary.")

    payload = {
        "model": chat_model,
        "messages": [{"role": "system", "content": system_message},
                     {"role": "user", "content": parts}],
        "temperature": 0.2,
        "max_tokens": MAX_CURATION_TOKENS,
        "response_format": {"type": "json_object"},
    }

    debug_dir = workspace.work / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    api_started = time.time()
    salvaged_clips = []
    salvaged_analysis = {}
    best_salvage = None
    final_result = None
    failure_summary = []
    last_content = ""

    for attempt in (1, 2, 3):
        if attempt == 2:
            # Same model, same task: continue from the unusable response.
            kept = ""
            if salvaged_clips:
                stamps = "\n".join(
                    json.dumps({"start_sec": c.get("start_sec"), "end_sec": c.get("end_sec")},
                               ensure_ascii=False)
                    for c in salvaged_clips)
                kept = (f"The selection below is FINAL - reuse these EXACT timestamps, never add, "
                        f"remove or shift a clip:\n{stamps}\n")
            va_note = ""
            if salvaged_analysis:
                va_note = ("Preserve this video analysis (you may shorten wording, never change facts):\n"
                           + json.dumps(salvaged_analysis, ensure_ascii=False)[:1200] + "\n")
            retry_text = (
                "Your previous response could not be used as-is. Return the COMPLETE JSON object "
                "with the exact schema from the original request. Remove unnecessary descriptive "
                "repetition so the whole object fits, but keep every mandatory field. "
                "Never invent clips, timestamps or quotes.\n\n" + kept + va_note)
            payload["messages"] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": parts},
                {"role": "assistant", "content": (last_content or "")[:6000]},
                {"role": "user", "content": retry_text},
            ]
            payload["temperature"] = 0.1
        elif attempt == 3:
            # Compact regeneration: same mandatory fields, terser values.
            kept = ""
            if salvaged_clips:
                stamps = "\n".join(
                    json.dumps({"start_sec": c.get("start_sec"), "end_sec": c.get("end_sec")},
                               ensure_ascii=False)
                    for c in salvaged_clips)
                kept = (f"The clip windows below are FINAL - reuse these EXACT timestamps, "
                        f"never add, remove or shift a clip:\n{stamps}\n")
            compact_prompt = (
                "Re-select the clips for this video using the attached keyframes and transcript, "
                "but with a COMPACT output structure so the complete JSON always fits:\n"
                "- video_analysis: topic + summary only (summary <= 200 chars).\n"
                "- clips: clip_index, start_sec, end_sec, viral_score (1-100), hook (verbatim or \"\"), "
                "title (<= 65 chars), description (<= 120 chars), tags (3-5), "
                "visual_analysis {main_subject, scene} (<= 60 chars each), "
                "zoom_intents [{start_rel_sec, end_rel_sec, zoom_level, target, reason <= 80 chars}] "
                "(empty array when no visual justification exists).\n"
                "Every mandatory field must be present for every clip. Return only JSON.\n\n" + kept)
            payload["messages"] = [
                {"role": "system", "content": "You return ONLY valid compact JSON, no Markdown fences."},
                {"role": "user", "content": [{"type": "text", "text": compact_prompt}] + parts[1:]},
            ]
            payload["temperature"] = 0.2

        info = gpt_client.chat_with_retries(settings.openrouter_api_base,
                                            settings.openrouter_api_key, payload,
                                            timeout=300, log=logger.info)
        last_content = info["content"] or ""
        data, parse_error = gpt_client.parse_json_content(last_content)
        parse_result = "ok" if data is not None else f"failed ({parse_error})"

        # A 200 response is never assumed complete: when the provider says the
        # output hit the limit, actively salvage what closed.
        if data is None and info["finish_reason"] == "length":
            data, repair_note = gpt_client.repair_truncated_json(last_content)
            if data is not None:
                parse_result += f"; salvaged ({repair_note})"
            else:
                parse_result += "; salvage failed"

        video_analysis, clips_raw, problems = gpt_client.validate_curation_schema(
            data, duration, low, high)
        validation_note = (f"{len(clips_raw)} clip(s) valid"
                           + (f"; problems: {'; '.join(problems[:6])}" if problems else ""))

        retry_reason = ""
        if info["ok"] and clips_raw:
            retry_reason = ""
        elif not info["ok"]:
            retry_reason = f"request failed: {info['error'][:200]}"
        elif info["finish_reason"] == "length":
            retry_reason = "finish_reason=length (output limit reached, response truncated)"
        elif info["finish_reason"] == "content_filter":
            retry_reason = "finish_reason=content_filter (response blocked by content filtering)"
        elif info["finish_reason"] not in ("stop", None):
            retry_reason = f"finish_reason={info['finish_reason']}"
        elif data is None:
            retry_reason = f"unparseable JSON ({parse_error[:160]})"
        else:
            retry_reason = "no clip passed schema validation"

        usage = info.get("usage") or {}
        logger.info("ATTEMPT %d/3 | MODEL: %s | HTTP STATUS: %d | FINISH REASON: %s | "
                    "PROMPT TOKENS: %s | COMPLETION TOKENS: %s | JSON VALIDATION: %s%s",
                    attempt, chat_model, info["http_status"], info["finish_reason"] or "(absent)",
                    usage.get("prompt_tokens") or "-", usage.get("completion_tokens") or "-",
                    validation_note, f" | RETRY REASON: {retry_reason}" if retry_reason else "")
        gpt_client.write_debug_attempt(
            debug_dir / f"curation_attempt_{attempt}.txt", f"{attempt}/3", chat_model,
            info, parse_result, validation_note, retry_reason, last_content)

        if info["ok"] and clips_raw and info["finish_reason"] in ("stop", None):
            # A complete response: parsed cleanly AND the provider confirmed the
            # output ended naturally (never assume a 200 alone means complete).
            final_result = (video_analysis, clips_raw)
            break

        failure_summary.append(f"attempt {attempt}: {retry_reason}")
        # Preserve whatever was recovered for the next attempt.
        if clips_raw:
            salvaged_clips = clips_raw
            if best_salvage is None or len(clips_raw) > len(best_salvage[1]):
                best_salvage = (video_analysis, clips_raw)
        if isinstance(video_analysis, dict) and video_analysis:
            salvaged_analysis = video_analysis

    if final_result is None and best_salvage is not None:
        # Every attempt ended truncated or unusable, but the model's own output
        # contained complete, individually validated clips. Using them verbatim
        # is recovery, not fabrication - strictly better than failing the job.
        logger.warning("all 3 GPT-6 Luna attempts ended incomplete; recovering %d complete "
                       "clip(s) verbatim from the model's own truncated output (validated, "
                       "never fabricated).", len(best_salvage[1]))
        final_result = best_salvage

    if final_result is None:
        raise CurationError(
            f"GPT-6 Luna curation failed after 3 attempts (model {chat_model}). "
            f"Failure summary: {'; '.join(failure_summary)}. Raw responses and diagnostics "
            "were saved to work/debug/curation_attempt_*.txt. Refusing to fabricate "
            "fallback clips - fix the API issue and re-run the job.")

    video_analysis, clips_raw = final_result
    api_seconds = round(time.time() - api_started, 1)

    (workspace.work / "video_analysis.json").write_text(
        json.dumps(video_analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    segments = transcript.get("segments", [])
    valid = []
    dropped = []
    for item in clips_raw:
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        length = end - start

        visual = item.get("visual_analysis") if isinstance(item.get("visual_analysis"), dict) else {}
        clip = {
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "viral_score": max(1, min(100, int(item.get("viral_score") or 50))),
            "topic": str(item.get("topic") or "").strip()[:120],
            "hook": str(item.get("hook") or "").strip()[:140],
            "title": str(item.get("title") or "").strip()[:65],
            "description": str(item.get("description") or "").strip()[:300],
            "tags": [],
            "visual_analysis": {
                "main_subject": str(visual.get("main_subject") or "").strip()[:160],
                "scene": str(visual.get("scene") or "").strip()[:160],
            },
            "zoom_intents": [],
        }
        clip["tags"] = _sanitize_tags(item.get("tags"))
        if not clip["title"] or not clip["description"]:
            dropped.append("missing title or description")
            continue
        if clip["hook"] and not _hook_is_real(clip["hook"], start, end, segments):
            logger.warning("hook for clip at %.0fs is not verbatim in the transcript; "
                           "dropping the quote.", start)
            clip["hook"] = ""

        intents = item.get("zoom_intents")
        if isinstance(intents, list):
            for intent in intents[:3]:
                if not isinstance(intent, dict):
                    continue
                try:
                    ws = max(0.0, float(intent.get("start_rel_sec", 0)))
                    we = min(length, float(intent.get("end_rel_sec", 0)))
                except (TypeError, ValueError):
                    continue
                if we - ws < 1.0:
                    continue
                try:
                    level = float(intent.get("zoom_level") or 1.10)
                except (TypeError, ValueError):
                    level = 1.10
                level = min(1.20, max(1.05, round(level, 2)))
                clip["zoom_intents"].append({
                    "start_rel_sec": round(ws, 2),
                    "end_rel_sec": round(we, 2),
                    "zoom_level": level,
                    "target": "primary_subject",
                    "reason": str(intent.get("reason") or "").strip()[:200],
                })
        valid.append(clip)

    # Remove heavily overlapping duplicates (keep the higher viral_score).
    valid.sort(key=lambda c: c["viral_score"], reverse=True)
    kept = []
    for clip in valid:
        duplicate = False
        for other in kept:
            overlap = min(clip["end_sec"], other["end_sec"]) - max(clip["start_sec"], other["start_sec"])
            shortest = min(clip["end_sec"] - clip["start_sec"], other["end_sec"] - other["start_sec"])
            if shortest > 0 and overlap / shortest > 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(clip)

    kept.sort(key=lambda c: c["start_sec"])
    for index, clip in enumerate(kept, 1):
        clip["clip_index"] = index

    if dropped:
        logger.warning("Dropped %d invalid AI clip entr%s: %s", len(dropped),
                       "y" if len(dropped) == 1 else "ies", dropped)
    if not kept:
        raise CurationError(
            "GPT-6 Luna produced no usable clips after validation. Raw response "
            "excerpts are in work/debug/curation_attempt_*.txt.")

    (workspace.work / "curation.json").write_text(
        json.dumps({"clips": kept}, ensure_ascii=False, indent=2), encoding="utf-8")

    cache_manager.save_cached(
        cache_paths.curation(),
        {"clips": kept, "video_analysis": video_analysis},
        meta={"source_sha256": source.get("source_sha256"), "model": chat_model,
              "target_duration": target_duration, "language_hint": language_hint,
              "prompt_version": CURATION_PROMPT_VERSION, "api_seconds": api_seconds})

    zoom_count = sum(len(c["zoom_intents"]) for c in kept)
    logger.info("Curation complete: %d clip(s) accepted, %d zoom intent(s) proposed "
                "(%.1fs, 1-3 GPT-6 Luna attempt(s); cached for this source hash).",
                len(kept), zoom_count, api_seconds)
    for clip in kept:
        logger.info("  clip %d: %s-%ss score=%d zoom_intents=%d title='%s'",
                    clip["clip_index"], clip["start_sec"], clip["end_sec"],
                    clip["viral_score"], len(clip["zoom_intents"]), clip["title"][:50])
    return kept, video_analysis


# ---------------------------------------------------------------------------
# Metadata refinement (PY_META)
# ---------------------------------------------------------------------------
def refine_metadata(clips: list, transcript: dict, tracks: dict, video_analysis: dict,
                    settings, workspace) -> list:
    chat_model = settings.openrouter_chat_model
    if not settings.openrouter_api_key:
        raise CurationError("OPENROUTER_API_KEY is not configured.")
    if chat_model != EXPECTED_CHAT_MODEL:
        raise CurationError(
            f"refusing to refine metadata with '{chat_model}'; Clipzy requires "
            f"'{EXPECTED_CHAT_MODEL}'.")

    def fmt_ts(seconds):
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"

    blocks = []
    for clip in clips:
        start, end = clip["start_sec"], clip["end_sec"]
        seg_lines = [f"[{fmt_ts(s['start'])}] {s['text']}" for s in transcript.get("segments", [])
                     if s["end"] > start - 10 and s["start"] < end + 10]
        context = "\n".join(seg_lines)[:3500] or "(no speech in this window)"
        faces_detected = bool((tracks.get(clip["clip_index"]) or {}).get("primary"))
        zoom_note = ", ".join(f"{z['start_rel_sec']}-{z['end_rel_sec']}s @ {z['zoom_level']}x"
                              for z in clip["editing"]["zoom_cuts"]) or "none"
        blocks.append(
            f"CLIP {clip['clip_index']} ({start:.1f}-{end:.1f}s) | faces_detected={faces_detected} | zoom: {zoom_note}\n"
            f"TRANSCRIPT CONTEXT:\n{context}")

    prompt = f"""You are refining the publishable metadata for clips you selected from a video. Use the exact transcript context below; the earlier multimodal analysis summary is also provided.

VIDEO ANALYSIS SUMMARY: {json.dumps(video_analysis, ensure_ascii=False)[:1500]}

SELECTED CLIPS:
{chr(10).join(blocks)}

RULES (per clip):
- title: HIGH-CTR & VIRAL, max 60 characters, written in the EXACT SPOKEN LANGUAGE of the video (e.g. Bahasa Indonesia for Indonesian videos). Specific, engaging, curiosity-inducing, never misleading. No generic filler ("Amazing Video", "Video Highlight", "Must Watch").
- description: what is actually said and shown in this specific clip in the video's spoken language, max 200 characters, do not repeat the title.
- topic: a short specific topic label in the video's spoken language.
- hook: a short quote VERBATIM from the transcript context inside the clip, or "" when none fits. Never invent quotes.
- tags: 4-5 niche-specific hashtags strictly grounded in the clip's topic, category, and keywords (e.g. #investasi #keuangan #saham for finance videos in Indonesian). Do NOT pad with generic #viral #fyp #shorts unless genuinely relevant.
- visual_analysis: main_subject (role-based description) and scene, grounded in the frame analysis.
- faces_detected is a measured computer-vision fact: never contradict it.

OUTPUT: a single JSON object {{"clips": [{{"clip_index": 1, "topic": "", "title": "", "description": "", "tags": ["#topic"], "hook": "", "visual_analysis": {{"main_subject": "", "scene": ""}}}}]}} with one entry per input clip. Return only JSON."""

    messages = [
        {"role": "system",
         "content": "You are Clipzy's metadata editor. You return ONLY valid JSON, never Markdown fences."},
        {"role": "user", "content": prompt},
    ]
    payload = {"model": chat_model, "messages": messages, "temperature": 0.3,
               "max_tokens": MAX_REFINE_TOKENS, "response_format": {"type": "json_object"}}

    debug_dir = workspace.work / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    refined = None
    last_content = ""
    best_refined = None
    for attempt in (1, 2, 3):
        if attempt == 2:
            payload["messages"] = [
                messages[0], messages[1],
                {"role": "assistant", "content": (last_content or "")[:6000]},
                {"role": "user", "content": (
                    "Your previous response could not be used. Return the COMPLETE JSON object "
                    "with the same schema and one entry per input clip. Remove verbose repetition "
                    "but keep every mandatory field. Never invent quotes or facts.")},
            ]
            payload["temperature"] = 0.2
        elif attempt == 3:
            payload["messages"] = [
                {"role": "system", "content": "You return ONLY valid compact JSON, no Markdown fences."},
                {"role": "user", "content": (
                    "Regenerate the metadata for the clips from the earlier prompt with a COMPACT "
                    "structure: title <= 65 chars, description <= 120 chars, topic <= 60 chars, "
                    "3-5 tags, hook verbatim or \"\", visual_analysis fields <= 60 chars. "
                    "One entry per input clip, clip_index must match. Return only JSON.")},
            ]
            payload["temperature"] = 0.3

        info = gpt_client.chat_with_retries(settings.openrouter_api_base,
                                            settings.openrouter_api_key, payload,
                                            timeout=240, log=logger.info)
        last_content = info["content"] or ""
        data, parse_error = gpt_client.parse_json_content(last_content)
        parse_result = "ok" if data is not None else f"failed ({parse_error})"
        if data is None and info["finish_reason"] == "length":
            data, repair_note = gpt_client.repair_truncated_json(last_content)
            if data is not None:
                parse_result += f"; salvaged ({repair_note})"

        refined_items = None
        validation_note = "no clips"
        if isinstance(data, dict) and isinstance(data.get("clips"), list):
            refined_items = {item["clip_index"]: item for item in data["clips"]
                             if isinstance(item, dict) and "clip_index" in item}
            validation_note = f"{len(refined_items)} clip entr{'y' if len(refined_items) == 1 else 'ies'}"

        retry_reason = ""
        if info["ok"] and refined_items:
            retry_reason = ""
        elif not info["ok"]:
            retry_reason = f"request failed: {info['error'][:200]}"
        elif info["finish_reason"] == "length":
            retry_reason = "finish_reason=length (response truncated)"
        elif info["finish_reason"] == "content_filter":
            retry_reason = "finish_reason=content_filter"
        elif data is None:
            retry_reason = f"unparseable JSON ({parse_error[:160]})"
        else:
            retry_reason = "response did not contain a clips array"

        usage = info.get("usage") or {}
        logger.info("REFINE ATTEMPT %d/3 | MODEL: %s | HTTP STATUS: %d | FINISH REASON: %s | "
                    "PROMPT TOKENS: %s | COMPLETION TOKENS: %s | JSON VALIDATION: %s%s",
                    attempt, chat_model, info["http_status"], info["finish_reason"] or "(absent)",
                    usage.get("prompt_tokens") or "-", usage.get("completion_tokens") or "-",
                    validation_note, f" | RETRY REASON: {retry_reason}" if retry_reason else "")
        gpt_client.write_debug_attempt(
            debug_dir / f"refine_attempt_{attempt}.txt", f"refine {attempt}/3", chat_model,
            info, parse_result, validation_note, retry_reason, last_content)

        if info["ok"] and refined_items and info["finish_reason"] in ("stop", None):
            refined = refined_items
            break
        if refined_items and (best_refined is None or len(refined_items) > len(best_refined)):
            best_refined = refined_items

    if refined is None and best_refined is not None:
        logger.warning("all 3 GPT-6 Luna refine attempts ended incomplete; using the %d "
                       "complete clip entr%s recovered verbatim from the model's own output.",
                       len(best_refined), "y" if len(best_refined) == 1 else "ies")
        refined = best_refined

    if refined is None:
        logger.warning("GPT-6 Luna metadata refinement failed after 3 attempts; keeping the "
                       "curation-stage metadata (also produced by GPT-6 Luna). No placeholder "
                       "metadata is fabricated. Diagnostics: work/debug/refine_attempt_*.txt.")
    else:
        segments = transcript.get("segments", [])
        for clip in clips:
            item = refined.get(clip["clip_index"])
            if not item:
                continue
            title = str(item.get("title") or "").strip()
            if title:
                clip["title"] = title[:65]
            description = str(item.get("description") or "").strip()
            if description:
                clip["description"] = description[:300]
            topic = str(item.get("topic") or "").strip()
            if topic:
                clip["topic"] = topic[:120]
            tags = item.get("tags")
            if isinstance(tags, list) and tags:
                clean = _sanitize_tags(tags)
                if clean:
                    clip["tags"] = clean[:5]
            hook = str(item.get("hook") or "").strip()
            if hook:
                start, end = clip["start_sec"], clip["end_sec"]
                window = " ".join(s["text"] for s in segments
                                  if s["end"] > start - 10 and s["start"] < end + 10).lower()
                probe = re.sub(r"\s+", " ", hook.lower()).strip(" .!?,")
                real = False
                if probe:
                    for pos in range(0, max(1, len(window) - len(probe) + 1)):
                        if SequenceMatcher(None, probe, window[pos:pos + len(probe)]).ratio() > 0.8:
                            real = True
                            break
                if real:
                    clip["hook"] = hook[:140]
                else:
                    logger.warning("refined hook for clip %d is not verbatim; dropped.",
                                   clip["clip_index"])
                    clip["hook"] = ""
            visual = item.get("visual_analysis") if isinstance(item.get("visual_analysis"), dict) else {}
            if visual.get("main_subject"):
                clip["visual_analysis"]["main_subject"] = str(visual["main_subject"]).strip()[:160]
            if visual.get("scene"):
                clip["visual_analysis"]["scene"] = str(visual["scene"]).strip()[:160]

    # Inject measured CV facts (never AI-guessed).
    for clip in clips:
        clip["visual_analysis"]["faces_detected"] = bool(
            (tracks.get(clip["clip_index"]) or {}).get("primary"))

    logger.info("Clip metadata ready (topic, hook, title, description, tags, visual analysis).")
    return clips
