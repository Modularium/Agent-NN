import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.agent_registry.routes import router, service
from services.agent_registry.schemas import AgentInfo
from core.agent_profile import AgentIdentity


@pytest.mark.unit
def test_agent_status_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(router)

    service._agents.clear()
    service._status.clear()

    service.register_agent(AgentInfo(name="writer_agent", url="http://w"))
    AgentIdentity(
        name="writer_agent",
        role="writer",
        traits={},
        skills=[],
        memory_index=None,
        created_at="now",
    ).save()

    client = TestClient(app)
    payload = {"last_response_duration": 2.0, "tasks_in_progress": 4}
    resp = client.post("/agent_status/writer_agent", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp = client.get("/agent_status/writer_agent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks_in_progress"] == 4
    assert data["avg_response_time"] == 2.0
    assert data["load_factor"] == 0.4

    profile = AgentIdentity.load("writer_agent")
    assert profile.avg_response_time == 2.0
    assert profile.load_factor == 0.4
