"""Shared GPT-6 Luna client helpers (port of the workflow's clipzy_gpt.py).

Protocol robustness lives here so both the curation and the metadata
refinement calls share one implementation:

* HTTP-level retries (transport errors, 429/5xx) with backoff;
* adaptive parameters (response_format / max_tokens parameter name) when the
  provider rejects them - never a model substitution;
* FULL response inspection: HTTP status, finish_reason, usage tokens and
  served model are captured and logged for every attempt;
* markdown-fence stripping and truncated-JSON repair that salvages only
  complete, syntactically closed elements (nothing is invented);
* structural schema validation of curation payloads;
* per-attempt debug artifacts (work/debug/*.txt) that never contain API keys,
  authorization headers or any other secret.
"""

from __future__ import annotations

import json
import os
import time

import requests

FATAL_STATUS = (401, 403, 404)


def chat_with_retries(api_base, api_key, payload, timeout=300, transport_retries=4, log=print):
    """POST /chat/completions with robust retry and adaptive parameters.

    Returns an info dict:
      ok            True when a 200 body yielded message.content
      http_status   last HTTP status (0 for transport errors)
      finish_reason choices[0].finish_reason (None when absent)
      content       choices[0].message.content (None when absent)
      usage         {prompt_tokens, completion_tokens, total_tokens}
      served_model  model reported by the provider
      error         last error text ("" when ok)

    `payload` MAY be mutated in place (response_format removal,
    max_tokens <-> max_completion_tokens) when the provider rejects those
    parameter names; each adaptation is applied at most once. A 200 status is
    never assumed to be a complete response: the caller must inspect
    finish_reason and parse the content.
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://clipzy.app",
        "X-Title": "Clipzy Engine",
    }
    info = {"ok": False, "http_status": 0, "finish_reason": None, "content": None,
            "usage": {}, "served_model": None, "error": ""}
    last_error = ""
    attempt = 0
    adapted = set()
    while attempt < transport_retries:
        attempt += 1
        if attempt > 1:
            delay = min(45, 8 * (2 ** (attempt - 2)))
            log(f"  HTTP retry {attempt}/{transport_retries} in {delay}s (last error: {last_error})")
            time.sleep(delay)
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"transport error: {exc}"
            log(f"  WARN: {last_error}")
            info["error"] = last_error
            continue
        info["http_status"] = response.status_code
        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError as exc:
                last_error = f"invalid JSON in 200 response: {exc}"
                log(f"  WARN: {last_error}")
                info["error"] = last_error
                continue
            choices = body.get("choices") or []
            if not choices:
                last_error = "empty choices array in 200 response"
                log(f"  WARN: {last_error}")
                info["error"] = last_error
                continue
            choice = choices[0]
            message = choice.get("message") or {}
            usage = body.get("usage") or {}
            info["content"] = message.get("content")
            info["finish_reason"] = choice.get("finish_reason")
            info["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            info["served_model"] = body.get("model")
            if info["content"] is None:
                last_error = "200 response has no message.content"
                log(f"  WARN: {last_error}")
                info["error"] = last_error
                continue
            info["ok"] = True
            info["error"] = ""
            return info
        body_text = (response.text or "")[:500]
        if response.status_code == 400:
            if ("response_format" in body_text and "response_format" in payload
                    and "response_format" not in adapted):
                adapted.add("response_format")
                payload.pop("response_format", None)
                log("  provider rejected response_format; retrying without it (same model).")
                attempt -= 1
                continue
            if ("max_tokens" in body_text and "max_tokens" in payload
                    and "token_param" not in adapted):
                adapted.add("token_param")
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                log("  provider prefers max_completion_tokens; retrying with that parameter name.")
                attempt -= 1
                continue
            if ("max_completion_tokens" in body_text and "max_completion_tokens" in payload
                    and "token_param" not in adapted):
                adapted.add("token_param")
                payload["max_tokens"] = payload.pop("max_completion_tokens")
                log("  provider prefers max_tokens; retrying with that parameter name.")
                attempt -= 1
                continue
        if response.status_code in FATAL_STATUS:
            info["error"] = f"HTTP {response.status_code}: {body_text}"
            log(f"  ERROR: request rejected (HTTP {response.status_code}): {body_text}")
            return info
        last_error = f"HTTP {response.status_code}: {body_text}"
        log(f"  WARN: {last_error}")
        info["error"] = last_error
    info["error"] = info["error"] or last_error or "exhausted transport retries"
    return info


def strip_code_fences(raw):
    """Remove Markdown code fences some models add despite json mode."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def parse_json_content(content):
    """Parse model output as JSON after fence stripping. -> (data, error)."""
    text = strip_code_fences(content)
    if not text:
        return None, "empty content"
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg} at line {exc.lineno} col {exc.colno}"


def _scan_structure(text):
    """Return (stack, cut_positions): positions right after a complete JSON
    element inside the root container, outside any string."""
    stack = []
    in_string = False
    escape = False
    cuts = []
    prev_structural = ""
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if stack and prev_structural == ":":
                    cuts.append(i + 1)  # end of a string VALUE
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append(ch)
            prev_structural = ch
        elif ch in "]}":
            if stack:
                stack.pop()
            prev_structural = ch
            if stack:
                cuts.append(i + 1)  # end of a nested object/array
        elif ch == ",":
            prev_structural = ","
        elif ch == ":":
            prev_structural = ":"
        elif not ch.isspace():
            prev_structural = ch
    return stack, cuts


def _close_brackets(prefix):
    """Append the closing brackets the prefix's open stack needs, first
    dropping a trailing comma. Best-effort, string-aware."""
    stack, _ = _scan_structure(prefix)
    text = prefix.rstrip()
    while text.endswith(","):
        text = text[:-1].rstrip()
        stack, _ = _scan_structure(text)
    closers = "".join("]" if ch == "[" else "}" for ch in reversed(stack))
    return text + closers


def repair_truncated_json(content):
    """Salvage complete objects from a truncated JSON document.

    Returns (data, note); data is None when nothing could be recovered. Only
    complete, syntactically closed elements survive the repair - the recovery
    NEVER invents or completes partial values itself.
    """
    text = strip_code_fences(content)
    if not text:
        return None, "empty content"
    try:
        return json.loads(text), "parsed without repair"
    except json.JSONDecodeError:
        pass
    _, cuts = _scan_structure(text)
    best = None
    for pos in reversed(cuts):
        candidate = _close_brackets(text[:pos])
        try:
            best = json.loads(candidate)
            return best, (f"recovered from a {len(text)}-char truncated response "
                          f"(kept the prefix ending at char {pos})")
        except json.JSONDecodeError:
            continue
    return None, "no complete element could be recovered"


def is_number(value):
    """True for real ints/floats (bools and NaN/inf excluded)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return value == value


def validate_curation_schema(data, duration, low, high):
    """Structural validation of a curation response.

    Returns (video_analysis, valid_raw_clips, problems). Clips that fail a
    REQUIRED check are rejected (listed in problems); nothing is fabricated to
    replace them. Timestamp/duration windows follow the same tolerance the
    pipeline has always applied.
    """
    problems = []
    if data is None:
        return {}, [], ["response is not JSON"]
    if isinstance(data, list):
        data = {"clips": data}
    if not isinstance(data, dict):
        return {}, [], ["root is not an object"]
    video_analysis = data.get("video_analysis")
    if not isinstance(video_analysis, dict):
        if "video_analysis" in data:
            problems.append("video_analysis is not an object")
        video_analysis = {}
    clips = data.get("clips")
    if not isinstance(clips, list):
        return video_analysis, [], ["clips is not an array"]
    if not clips:
        return video_analysis, [], ["clips array is empty"]
    valid = []
    for i, item in enumerate(clips):
        tag = f"clip[{i}]"
        if not isinstance(item, dict):
            problems.append(f"{tag}: not an object")
            continue
        required = ("start_sec", "end_sec", "viral_score", "title",
                    "description", "tags", "visual_analysis", "zoom_intents")
        missing = [field for field in required if field not in item]
        if missing:
            problems.append(f"{tag}: missing {', '.join(missing)}")
            continue
        start, end = item["start_sec"], item["end_sec"]
        if not (is_number(start) and is_number(end)):
            problems.append(f"{tag}: non-numeric timestamps")
            continue
        start, end = float(start), float(end)
        length = end - start
        if start < 0 or end > duration or length < 3.0:
            problems.append(f"{tag}: window {start:.1f}-{end:.1f}s invalid for a {duration:.1f}s source")
            continue
        if duration >= low and not (max(3, low - 5) <= length <= min(duration, high + 5)):
            problems.append(f"{tag}: length {length:.1f}s outside requested {low}-{high}s")
            continue
        if not is_number(item["viral_score"]):
            problems.append(f"{tag}: viral_score {item['viral_score']!r} is not a number")
            continue
        if not isinstance(item["tags"], list):
            problems.append(f"{tag}: tags is not an array")
            continue
        if not isinstance(item["visual_analysis"], dict):
            problems.append(f"{tag}: visual_analysis is not an object")
            continue
        if not isinstance(item["zoom_intents"], list):
            problems.append(f"{tag}: zoom_intents is not an array")
            continue
        if not str(item.get("title") or "").strip():
            problems.append(f"{tag}: empty title")
            continue
        if not str(item.get("description") or "").strip():
            problems.append(f"{tag}: empty description")
            continue
        valid.append(item)
    return video_analysis, valid, problems


def write_debug_attempt(path, label, model, info, parse_result,
                        validation_note, retry_reason, content):
    """Persist one attempt's diagnostics. NEVER writes secrets."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    usage = (info or {}).get("usage") or {}
    lines = [
        f"ATTEMPT: {label}",
        f"MODEL: {model}",
        f"SERVED MODEL: {(info or {}).get('served_model') or '(unreported)'}",
        f"HTTP STATUS: {(info or {}).get('http_status', 0)}",
        f"FINISH REASON: {(info or {}).get('finish_reason') or '(absent)'}",
        f"PROMPT TOKENS: {usage.get('prompt_tokens') or '(unreported)'}",
        f"COMPLETION TOKENS: {usage.get('completion_tokens') or '(unreported)'}",
        f"TOTAL TOKENS: {usage.get('total_tokens') or '(unreported)'}",
        f"RESPONSE LENGTH (chars): {len(content or '')}",
        f"JSON PARSING: {parse_result}",
        f"JSON VALIDATION: {validation_note}",
        f"RETRY REASON: {retry_reason or '-'}",
        "",
        "----- RESPONSE CONTENT (first 20000 chars) -----",
        (content or "")[:20000],
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
