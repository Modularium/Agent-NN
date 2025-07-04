from fastapi.testclient import TestClient
import pytest
import sys
import types

sys.modules.setdefault(
    "mlflow",
    types.SimpleNamespace(
        start_run=lambda *a, **k: None,
        set_tag=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        set_tracking_uri=lambda *a, **k: None,
        tracking=types.SimpleNamespace(
            MlflowClient=lambda: types.SimpleNamespace(
                list_experiments=lambda: [],
                get_run=lambda run_id: types.SimpleNamespace(
                    info=types.SimpleNamespace(run_id=run_id, status="FINISHED"),
                    data=types.SimpleNamespace(metrics={}, params={}),
                ),
            )
        ),
    ),
)

from agentnn.mcp import mcp_server  # noqa: E402


class DummyConnector:
    def __init__(self, *a, **k):
        pass

    async def get(self, path: str):
        if path == "/agents":
            return []
        return {}

    async def post(self, *a, **k):
        return {}


@pytest.mark.unit
def test_server_routes(monkeypatch):
    monkeypatch.setattr(mcp_server, "ServiceConnector", DummyConnector)
    app = mcp_server.create_app()
    client = TestClient(app)
    assert client.get("/v1/mcp/ping").status_code == 200
    assert client.get("/v1/mcp/context/map").status_code == 200
    assert client.get("/v1/mcp/agent/list").status_code == 200
