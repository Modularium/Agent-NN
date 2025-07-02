from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.endpoints import APIEndpoints


class DummyAgent:
    def __init__(self):
        self.called = False

    def get_config(self):
        return {
            "name": "demo",
            "domain": "demo",
            "tools": [],
            "model_config": {"model": "gpt"},
            "created_at": "2024-01-01",
            "version": "1.0.0",
        }

    def get_status(self):
        return {
            "agent_id": "demo",
            "name": "demo",
            "domain": "demo",
            "status": "active",
            "capabilities": [],
            "total_tasks": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "last_active": "2024-01-01",
        }


def create_client(agent):
    api = APIEndpoints()
    api.agent_manager.get_agent = lambda _id: agent
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def test_flowise_export_success():
    client = create_client(DummyAgent())
    resp = client.get("/agents/demo?format=flowise")
    assert resp.status_code == 200
    assert resp.json()["id"] == "demo"


def test_flowise_export_not_found():
    client = create_client(None)
    resp = client.get("/agents/unknown?format=flowise")
    assert resp.status_code == 404


class NoConfigAgent(DummyAgent):
    def get_config(self):
        raise AttributeError


def test_flowise_export_unsupported():
    client = create_client(NoConfigAgent())
    resp = client.get("/agents/demo?format=flowise")
    assert resp.status_code == 400
