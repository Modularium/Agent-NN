import importlib
from fastapi.testclient import TestClient
import pytest

SERVICES = [
    "services.agent_registry.main",
    "services.session_manager.main",
    "services.coalition_manager.main",
    "services.llm_gateway.main",
]

@pytest.mark.parametrize("module_name", SERVICES)
def test_metrics_route_available(module_name, monkeypatch):
    if module_name == "services.llm_gateway.main":
        monkeypatch.setattr("services.llm_gateway.service.pipeline", None)
        monkeypatch.setattr("services.llm_gateway.service.SentenceTransformer", None)
    module = importlib.import_module(module_name)
    app = getattr(module, "app")
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "agentnn_response_seconds" in resp.text
