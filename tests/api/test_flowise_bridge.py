import sys
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.modules.setdefault(
    "mlflow",
    SimpleNamespace(
        start_run=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
        log_metrics=lambda *a, **k: None,
        set_tags=lambda *a, **k: None,
        active_run=lambda: None,
    ),
)  # noqa: E402
sys.modules.setdefault("numpy", SimpleNamespace(array=lambda *a, **k: None))  # noqa: E402
sys.modules.setdefault("torch", SimpleNamespace(Tensor=object))  # noqa: E402

from api.flowise_bridge import FlowiseBridge  # noqa: E402

import pytest  # noqa: E402


@pytest.mark.unit
def test_run_task(monkeypatch):
    async def fake_request(self, method, path, payload=None):
        assert method == "POST"
        assert path == "/tasks"
        return {"task_id": "1", "status": "queued"}

    monkeypatch.setattr(FlowiseBridge, "_request", fake_request)
    bridge = FlowiseBridge("http://api")
    app = FastAPI()
    app.include_router(bridge.router)
    client = TestClient(app)

    resp = client.post("/flowise/run_task", json={"description": "hi", "domain": None})
    assert resp.status_code == 200
    assert resp.json()["result"]["task_id"] == "1"


@pytest.mark.unit
def test_get_status(monkeypatch):
    async def fake_request(self, method, path, payload=None):
        assert method == "GET"
        assert path == "/tasks/1"
        return {"task_id": "1", "result": "ok"}

    monkeypatch.setattr(FlowiseBridge, "_request", fake_request)
    bridge = FlowiseBridge("http://api")
    app = FastAPI()
    app.include_router(bridge.router)
    client = TestClient(app)

    resp = client.get("/flowise/status/1")
    assert resp.status_code == 200
    assert resp.json()["result"] == "ok"
