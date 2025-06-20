import json
import urllib.request

from mcp.worker_dev.service import WorkerService


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def fake_urlopen(req, timeout=0):
    data = json.loads(req.data.decode())
    return FakeResponse({"text": f"generated {data['prompt']}"})


def test_worker_dev_execute(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    service = WorkerService()
    result = service.execute_task("prints hello")
    assert "generated" in result
