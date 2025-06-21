import os
import importlib
from fastapi.testclient import TestClient
from core.model_context import ModelContext, TaskContext
from services.task_dispatcher import routes, main


def _setup(monkeypatch) -> TestClient:
    os.environ["AUTH_ENABLED"] = "true"
    os.environ["API_TOKENS"] = "valid"
    os.environ["RATE_LIMITS_ENABLED"] = "false"
    os.environ["INPUT_LIMIT_BYTES"] = "10"
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


def test_input_validation(monkeypatch):
    client = _setup(monkeypatch)
    headers = {"Authorization": "Bearer valid"}
    payload = {"task_type": "demo", "input_data": {"text": "x" * 20}}
    resp = client.post("/task", json=payload, headers=headers)
    assert resp.status_code == 422
