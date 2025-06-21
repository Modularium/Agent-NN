import os
import importlib
from fastapi.testclient import TestClient
from core.model_context import ModelContext, TaskContext
from services.task_dispatcher import routes, main


def _setup(monkeypatch) -> TestClient:
    os.environ["AUTH_ENABLED"] = "true"
    os.environ["API_TOKENS"] = "valid"
    os.environ["RATE_LIMITS_ENABLED"] = "true"
    os.environ["RATE_LIMIT_TASK"] = "2/minute"
    importlib.reload(main)
    importlib.reload(routes)
    monkeypatch.setattr(
        routes.service,
        "dispatch_task",
        lambda task, session_id=None: ModelContext(
            task=task.task_id, task_context=task
        ),
    )
    return TestClient(main.app)


def test_rate_limit(monkeypatch):
    client = _setup(monkeypatch)
    headers = {"Authorization": "Bearer valid"}
    for _ in range(2):
        resp = client.post("/task", json={"task_type": "demo"}, headers=headers)
        assert resp.status_code == 200
    resp = client.post("/task", json={"task_type": "demo"}, headers=headers)
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
