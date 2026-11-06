"""Job request schema.

The wire format is the JSON documented in docs/API.md; it accepts both the
snake_case keys (Android payload / repository_dispatch) and the camelCase
aliases the original workflow accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from utils.validation import validate_job_request


@dataclass
class JobRequest:
    params: dict
    raw: dict = field(default_factory=dict)

    @property
    def job_id(self) -> str:
        return self.params["job_id"]

    @property
    def youtube_url(self) -> str:
        return self.params["youtube_url"]

    @classmethod
    def from_payload(cls, payload: dict) -> "JobRequest":
        params, errors = validate_job_request(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(params=params, raw=dict(payload))

    def summary(self) -> dict:
        shown = dict(self.params)
        if shown.get("youtube_url"):
            shown["youtube_url"] = shown["youtube_url"][:80]
        return shown


def parse_job_payload(body) -> tuple[JobRequest | None, list[str]]:
    """Parse a request body (dict or JSON string) into a JobRequest.

    Returns (request, errors); errors is non-empty when the payload is invalid.
    """
    if isinstance(body, (str, bytes)):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, ["request body is not valid JSON."]
    if not isinstance(body, dict):
        return None, ["request body must be a JSON object."]
    payload = dict(body.get("payload") if isinstance(body.get("payload"), dict) else body)
    try:
        return JobRequest.from_payload(payload), []
    except ValueError as exc:
        return None, [str(exc)]
