import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.agent_registry.routes import router, service
from services.agent_registry.service import AgentRegistryService
from core.metrics_utils import TASKS_PROCESSED


@pytest.mark.unit
def test_service_get_missing_agent_metrics():
    svc = AgentRegistryService()
    start = TASKS_PROCESSED.labels("agent_registry")._value.get()
    assert svc.get_agent("missing") is None
    assert TASKS_PROCESSED.labels("agent_registry")._value.get() == start + 1


@pytest.mark.unit
def test_get_agent_route_not_found():
    app = FastAPI()
    app.include_router(router)

    service._agents.clear()
    client = TestClient(app)
    resp = client.get("/agents/unknown")
    assert resp.status_code == 404
