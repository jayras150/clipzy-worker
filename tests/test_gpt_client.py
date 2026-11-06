"""GPT client protocol tests: retries, adaptation, JSON repair, validation."""

import json

import pytest
import requests

from engines import gpt_client


class FakeResponse:
    def __init__(self, status_code=200, text="", body=None):
        self.status_code = status_code
        self.text = text
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _ok_body(content="hello", finish="stop"):
    return {"choices": [{"message": {"content": content},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "openai/gpt-6-luna"}


def test_chat_with_retries_success(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return FakeResponse(body=_ok_body('{"ok": true}'))

    monkeypatch.setattr(requests, "post", fake_post)
    payload = {"model": "openai/gpt-6-luna", "max_tokens": 10}
    info = gpt_client.chat_with_retries("http://x", "key", payload, log=lambda *_: None)
    assert info["ok"] is True
    assert info["content"] == '{"ok": true}'
    assert info["served_model"] == "openai/gpt-6-luna"
    assert info["usage"]["total_tokens"] == 15
    assert len(calls) == 1


def test_chat_with_retries_transport_retry_then_success(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("boom")
        return FakeResponse(body=_ok_body("ok"))

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(gpt_client.time, "sleep", lambda s: None)
    info = gpt_client.chat_with_retries("http://x", "key", {}, log=lambda *_: None)
    assert info["ok"] is True
    assert len(calls) == 2


def test_chat_with_retries_fatal_status_no_retry(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return FakeResponse(status_code=401, text="unauthorized")

    monkeypatch.setattr(requests, "post", fake_post)
    info = gpt_client.chat_with_retries("http://x", "key", {}, log=lambda *_: None)
    assert info["ok"] is False
    assert info["http_status"] == 401
    assert len(calls) == 1


def test_chat_with_retries_adapts_response_format(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            return FakeResponse(status_code=400, text="response_format not supported")
        return FakeResponse(body=_ok_body("{}"))

    monkeypatch.setattr(requests, "post", fake_post)
    payload = {"model": "m", "response_format": {"type": "json_object"}}
    info = gpt_client.chat_with_retries("http://x", "key", payload, log=lambda *_: None)
    assert info["ok"] is True
    assert "response_format" not in payload  # removed before retry
    assert len(calls) == 2


def test_chat_with_retries_adapts_token_param_name(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if "max_tokens" in json:
            return FakeResponse(status_code=400,
                                text="Parameter 'max_tokens' unsupported; use max_completion_tokens")
        return FakeResponse(body=_ok_body("{}"))

    monkeypatch.setattr(requests, "post", fake_post)
    payload = {"model": "m", "max_tokens": 5}
    info = gpt_client.chat_with_retries("http://x", "key", payload, log=lambda *_: None)
    assert info["ok"] is True
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 5


# ---------------------------------------------------------------------------
# JSON parsing / repair
# ---------------------------------------------------------------------------
def test_strip_code_fences():
    assert gpt_client.strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert gpt_client.strip_code_fences('```\n{"a": 1}') == '{"a": 1}'
    assert gpt_client.strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_parse_json_content():
    assert gpt_client.parse_json_content('{"a": 1}') == ({"a": 1}, "")
    data, err = gpt_client.parse_json_content("```json\n{\"a\": 1}\n```")
    assert data == {"a": 1} and err == ""
    data, err = gpt_client.parse_json_content("")
    assert data is None and "empty" in err


def test_repair_truncated_json_recovers_complete_objects():
    truncated = ('{"video_analysis": {"topic": "t"}, "clips": ['
                 '{"clip_index": 1, "start_sec": 0.0, "end_sec": 5.0, "viral_score": 80, '
                 '"title": "A", "description": "d", "tags": ["#x"], '
                 '"visual_analysis": {"main_subject": "s", "scene": "c"}, '
                 '"zoom_intents": []}, {"clip_index": 2, "start_sec": 10.0, "end_')
    data, note = gpt_client.repair_truncated_json(truncated)
    assert data is not None
    assert len(data["clips"]) == 1  # only the COMPLETE clip survives
    assert data["clips"][0]["clip_index"] == 1
    assert "recovered" in note


def test_repair_truncated_json_unrecoverable():
    data, note = gpt_client.repair_truncated_json('{"clips": [{"start')
    assert data is None


def test_is_number():
    assert gpt_client.is_number(5) and gpt_client.is_number(5.5)
    assert not gpt_client.is_number(True)
    assert not gpt_client.is_number("5")
    assert not gpt_client.is_number(float("nan"))


# ---------------------------------------------------------------------------
# Curation schema validation
# ---------------------------------------------------------------------------
def _clip(**overrides):
    base = {"start_sec": 0.0, "end_sec": 30.0, "viral_score": 50, "title": "t",
            "description": "d", "tags": ["#a"], "visual_analysis": {},
            "zoom_intents": []}
    base.update(overrides)
    return base


def test_validate_curation_schema_happy_path():
    data = {"video_analysis": {"topic": "x"},
            "clips": [_clip(), _clip(start_sec=40.0, end_sec=80.0)]}
    va, clips, problems = gpt_client.validate_curation_schema(data, 100.0, 30, 60)
    assert not problems
    assert len(clips) == 2
    assert va["topic"] == "x"


def test_validate_curation_schema_rejects_bad_windows():
    data = {"clips": [_clip(end_sec=150.0)]}
    _va, clips, problems = gpt_client.validate_curation_schema(data, 100.0, 30, 60)
    assert not clips
    assert any("invalid for a" in p for p in problems)

    data = {"clips": [_clip(start_sec=0, end_sec=10)]}  # too short vs 30-60 target
    _va, clips, problems = gpt_client.validate_curation_schema(data, 100.0, 30, 60)
    assert not clips
    assert any("outside requested" in p for p in problems)


def test_validate_curation_schema_missing_required_fields():
    clip = _clip()
    del clip["zoom_intents"]
    _va, clips, problems = gpt_client.validate_curation_schema({"clips": [clip]}, 100.0, 30, 60)
    assert not clips and any("missing" in p for p in problems)


def test_validate_curation_schema_list_root_and_empty():
    va, clips, problems = gpt_client.validate_curation_schema([_clip()], 100.0, 30, 60)
    assert len(clips) == 1
    _va, clips, problems = gpt_client.validate_curation_schema({"clips": []}, 100.0, 30, 60)
    assert not clips and "empty" in problems[0]
    _va, clips, problems = gpt_client.validate_curation_schema(None, 100.0, 30, 60)
    assert not clips and "not JSON" in problems[0]


def test_write_debug_attempt_never_writes_secrets(tmp_path):
    path = tmp_path / "debug" / "attempt_1.txt"
    info = {"usage": {"total_tokens": 5}, "http_status": 200, "finish_reason": "stop",
            "served_model": "m"}
    gpt_client.write_debug_attempt(path, "1/3", "m", info, "ok", "1 clip(s)", "", "content")
    text = path.read_text()
    assert "Bearer" not in text
    assert "ATTEMPT: 1/3" in text
