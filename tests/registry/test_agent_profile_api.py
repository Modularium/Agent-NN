from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.agent_registry.routes import router, service
from services.agent_registry.schemas import AgentInfo


def test_agent_profile_api(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    service.register_agent(AgentInfo(name="writer_agent", url="http://w"))
    client = TestClient(app)

    r = client.get("/agent_profile/writer_agent")
    assert r.status_code == 200
    assert r.json()["name"] == "writer_agent"

    r = client.post("/agent_profile/writer_agent", json={"traits": {"x": 1}})
    assert r.status_code == 200
    assert r.json()["traits"]["x"] == 1

    r = client.get("/agents")
    assert r.status_code == 200
    data = r.json()
    assert data["agents"][0]["traits"] is not None
