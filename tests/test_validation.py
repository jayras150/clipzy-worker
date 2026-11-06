"""Request validation and normalization tests (port of PY_NORMALIZE)."""

import pytest

from schemas.job_request import JobRequest, parse_job_payload
from utils.validation import validate_job_request

VALID = {
    "job_id": "clipzy_001",
    "youtube_url": "https://www.youtube.com/watch?v=example",
    "target_duration": "30-60",
    "subtitle_style": "VIRAL",
    "subtitle_language": "",
    "aspect_ratio": "source",
    "preserve_source_quality": True,
    "output_quality": "original",
    "manual_transform": None,
    "watermark_base64": "",
    "watermark_url": "",
    "watermark_position": "bottom_right",
    "watermark_size": "medium",
}


def test_valid_request_defaults():
    request = JobRequest.from_payload({"youtube_url": "https://youtu.be/x"})
    assert request.params["job_id"] == "test_job"
    assert request.params["target_duration"] == "30-60"
    assert request.params["subtitle_style"] == "VIRAL"
    assert request.params["aspect_ratio"] == "source"
    assert request.params["preserve_source_quality"] == "true"
    assert request.params["output_quality"] == "original"
    assert request.params["watermark_position"] == "bottom_right"
    assert request.params["watermark_size"] == "medium"


def test_camelcase_aliases():
    request = JobRequest.from_payload({
        "youtubeUrl": "https://youtu.be/x",
        "jobId": "android_job",
        "subtitleStyle": "neon",
        "aspectRatio": "9:16",
        "outputQuality": "720p",
        "preserveSourceQuality": "false",
    })
    assert request.params["job_id"] == "android_job"
    assert request.params["subtitle_style"] == "NEON"
    assert request.params["aspect_ratio"] == "9:16"
    assert request.params["output_quality"] == "720p"
    assert request.params["preserve_source_quality"] == "false"


@pytest.mark.parametrize("overrides,expected_fragment", [
    ({"youtube_url": ""}, "youtube_url is empty"),
    ({"youtube_url": "ftp://bad"}, "must start with http"),
    ({"youtube_url": "https://a b.com"}, "must not contain whitespace"),
    ({"job_id": "bad job!"}, "job_id may contain only"),
    ({"target_duration": "abc"}, "target_duration must be a number"),
    ({"subtitle_style": "COMIC"}, "subtitle_style must be one of"),
    ({"subtitle_language": "english"}, "BCP-47"),
    ({"aspect_ratio": "5:4"}, "aspect_ratio must be one of"),
    ({"output_quality": "4k"}, "output_quality must be one of"),
    ({"preserve_source_quality": "maybe"}, "preserve_source_quality must be"),
    ({"manual_transform": "not json {"}, "manual_transform is not valid JSON"),
    ({"manual_transform": [1, 2]}, "manual_transform must be an object"),
])
def test_invalid_requests(overrides, expected_fragment):
    payload = {**VALID, **overrides}
    _params, errors = validate_job_request(payload)
    assert errors, f"expected error for {overrides}"
    assert any(expected_fragment in e for e in errors)


def test_manual_transform_clamping():
    payload = {**VALID, "manual_transform": {"scale": 9.0, "offset_x": 0.9, "offset_y": -0.9}}
    params, errors = validate_job_request(payload)
    assert not errors
    manual = params["manual_transform"]
    assert manual["scale"] == 5.0
    limit = (1.0 - 1.0 / 5.0) / 2.0
    assert manual["offset_x"] == round(limit, 4)
    assert manual["offset_y"] == round(-limit, 4)


def test_manual_transform_identity_dropped():
    payload = {**VALID, "manual_transform": {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0}}
    params, _ = validate_job_request(payload)
    assert params["manual_transform"] is None


def test_manual_transform_string_json_and_camel():
    payload = {**VALID, "manualTransform": '{"scale": 1.5, "offsetX": 0.1}'}
    params, _ = validate_job_request(payload)
    assert params["manual_transform"]["scale"] == 1.5
    assert params["manual_transform"]["offset_x"] == 0.1


def test_watermark_position_and_size_normalized():
    payload = {**VALID, "watermark_position": "CENTER", "watermark_size": "HUGE"}
    params, _ = validate_job_request(payload)
    assert params["watermark_position"] == "bottom_right"
    assert params["watermark_size"] == "medium"


def test_parse_job_payload_envelope_and_errors():
    request, errors = parse_job_payload({"payload": {**VALID}})
    assert request is not None and not errors
    request, errors = parse_job_payload("{not json")
    assert request is None and errors
    request, errors = parse_job_payload([1, 2, 3])
    assert request is None and errors
    request, errors = parse_job_payload({**VALID, "job_id": "bad id!"})
    assert request is None and errors
