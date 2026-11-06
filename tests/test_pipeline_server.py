"""Pipeline HTTP server + job manager tests (aiohttp test client)."""

import asyncio
import json
import time

import pytest
import pytest_asyncio

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from config import WorkerSettings  # noqa: E402
from schemas.job_request import JobRequest  # noqa: E402
from services.job_manager import (COMPLETED, FAILED, JobManager)  # noqa: E402
from services.pipeline_server import build_app  # noqa: E402


VALID_BODY = {
    "job_id": "srv_001",
    "youtube_url": "https://www.youtube.com/watch?v=x",
    "target_duration": "30-60",
    "subtitle_style": "VIRAL",
    "aspect_ratio": "source",
    "output_quality": "original",
}


@pytest.fixture()
def stub_manager(monkeypatch):
    """JobManager with the pipeline replaced by a controllable stub."""
    from pipeline import runner as runner_mod
    from services import job_manager as jm_mod

    outcomes = {}

    def stub_run(request, settings, progress, log, cancel_check=None, r2=None):
        outcome = outcomes.get(request.job_id, {"status": "completed"})
        if outcome["status"] == "fail":
            raise runner_mod.JobFailure(outcome["error"])
        for stage in ("download", "render"):
            progress.stage(stage)
        return {"job_id": request.job_id, "status": "completed",
                "progress": 100, "clip_count": 0, "clips": [],
                "download_url": "", "manifest_url": "",
                "processing_time": 1.0, "error": None}

    monkeypatch.setattr(runner_mod, "run_pipeline", stub_run)
    monkeypatch.setattr(jm_mod, "run_pipeline", stub_run)
    settings = WorkerSettings()
    manager = JobManager(settings, r2=None)
    manager.settings = settings
    manager.outcomes = outcomes
    return manager


@pytest_asyncio.fixture()
async def client(stub_manager):
    app = build_app(stub_manager.settings, stub_manager)
    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_run_accepts_valid_job(client, stub_manager):
    response = await client.post("/run", json=VALID_BODY)
    assert response.status == 202
    body = await response.json()
    assert body["ok"] is True
    assert body["job_id"] == "srv_001"

    await asyncio.sleep(0.3)  # let the worker thread finish
    status = await client.post("/status", json={"job_id": "srv_001"})
    assert status.status == 200
    doc = await status.json()
    assert doc["status"] == COMPLETED
    assert doc["response"]["job_id"] == "srv_001"


@pytest.mark.asyncio
async def test_run_rejects_invalid_payload(client):
    response = await client.post("/run", json={"youtube_url": ""})
    assert response.status == 422
    body = await response.json()
    assert "youtube_url is empty" in body["error"]

    response = await client.post("/run", data=b"{not json")
    assert response.status == 400

    response = await client.post("/run", json=[1, 2])
    assert response.status == 400


@pytest.mark.asyncio
async def test_status_unknown_job(client):
    response = await client.post("/status", json={"job_id": "nope"})
    assert response.status == 404


@pytest.mark.asyncio
async def test_run_conflict_for_duplicate_job_id(client, stub_manager):
    stub_manager.outcomes["dup_001"] = {"status": "slow"}
    stub_manager.settings.max_concurrent_jobs = 4  # isolate the duplicate check
    # Make the stub block until we release it.
    import threading

    release = threading.Event()

    from services import job_manager as jm_mod

    def blocking_stub(request, *args, **kwargs):
        release.wait(timeout=5)
        return {"job_id": request.job_id, "status": "completed"}

    jm_mod.run_pipeline = blocking_stub
    try:
        first = await client.post("/run", json={**VALID_BODY, "job_id": "dup_001"})
        assert first.status == 202
        second = await client.post("/run", json={**VALID_BODY, "job_id": "dup_001"})
        assert second.status == 409
        release.set()
        await asyncio.sleep(0.3)
    finally:
        del jm_mod.run_pipeline


@pytest.mark.asyncio
async def test_cancel_stops_job(client, stub_manager):
    import threading

    release = threading.Event()
    from pipeline import runner as runner_mod
    from services import job_manager as jm_mod

    def cancellable_stub(request, settings, progress, log, cancel_check=None, r2=None):
        while not release.is_set() and not (cancel_check and cancel_check()):
            time.sleep(0.05)
        if cancel_check and cancel_check():
            raise runner_mod.JobCancelled("cancelled by request")
        return {"job_id": request.job_id, "status": "completed"}

    jm_mod.run_pipeline = cancellable_stub
    try:
        response = await client.post("/run", json={**VALID_BODY, "job_id": "cancel_001"})
        assert response.status == 202
        await asyncio.sleep(0.2)
        cancel = await client.post("/cancel", json={"job_id": "cancel_001"})
        body = await cancel.json()
        assert body["cancelled"] is True
        await asyncio.sleep(0.3)
        status = await client.post("/status", json={"job_id": "cancel_001"})
        doc = await status.json()
        assert doc["status"] == "cancelled"
    finally:
        release.set()
        del jm_mod.run_pipeline


@pytest.mark.asyncio
async def test_probe_and_health(client):
    probe = await client.post("/probe", json={})
    assert probe.status == 200
    doc = await probe.json()
    assert doc["ok"] is True
    assert "gpu" in doc

    health = await client.get("/health")
    assert health.status == 200
    assert (await health.json())["ok"] is True


@pytest.mark.asyncio
async def test_failed_job_reports_error(client, stub_manager):
    stub_manager.outcomes["fail_001"] = {"status": "fail",
                                         "error": "simulated pipeline failure"}
    response = await client.post("/run", json={**VALID_BODY, "job_id": "fail_001"})
    assert response.status == 202
    await asyncio.sleep(0.3)
    status = await client.post("/status", json={"job_id": "fail_001"})
    doc = await status.json()
    assert doc["status"] == FAILED
    assert "simulated pipeline failure" in doc["error"]


@pytest.mark.asyncio
async def test_envelope_payload_wrapping(client, stub_manager):
    """The Vast.ai envelope nests the user payload under 'payload'."""
    response = await client.post("/run", json={"payload": VALID_BODY})
    assert response.status == 202
    body = await response.json()
    assert body["job_id"] == "srv_001"
