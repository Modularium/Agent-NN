import json
import urllib.request
from types import SimpleNamespace

from mcp.task_dispatcher.service import TaskDispatcherService


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
    return FakeResponse({"result": "ok"})


def test_dispatch_no_agent():
    service = TaskDispatcherService()
    service.registry = SimpleNamespace(list_agents=lambda: [])
    result = service.dispatch_task("x", "y", None)
    assert "error" in result


def test_dispatch_worker(monkeypatch):
    service = TaskDispatcherService()
    service.registry = SimpleNamespace(list_agents=lambda: [{"name": "worker_dev", "url": "http://w"}])
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = service.dispatch_task("greeting", "hello", None)
    assert result["response"]["result"] == "ok"
