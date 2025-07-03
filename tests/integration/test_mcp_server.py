from fastapi.testclient import TestClient
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "mcp_server", pathlib.Path("agentnn/mcp/mcp_server.py")
)
mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server)  # type: ignore


class DummyConnector:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    async def post(self, path: str, json: dict) -> dict:
        return {"echo": path, **json}

    async def get(self, path: str) -> dict:
        return {"path": path, "context": []}


def test_execute_roundtrip(monkeypatch):
    monkeypatch.setattr(mcp_server, "ServiceConnector", DummyConnector)
    app = mcp_server.create_app()
    client = TestClient(app)

    ctx = {"task": "demo"}
    resp = client.post("/v1/mcp/execute", json=ctx)
    assert resp.status_code == 200
    data = resp.json()
    assert data["echo"] == "/dispatch"

    resp = client.post("/v1/mcp/context", json=ctx)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp = client.get("/v1/mcp/context/123")
    assert resp.status_code == 200
    assert resp.json()["path"] == "/context/123"
