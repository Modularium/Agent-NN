import urllib.request

from mcp.task_dispatcher.service import TaskDispatcherService


def test_legacy_task_type(monkeypatch):
    service = TaskDispatcherService()
    service.registry.list_agents = lambda: [{"name": "worker_dev", "url": "http://w"}]
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=10: DummyResponse({"text": "ok"})
    )
    result = service.dispatch_task("say_hello", "world", None)
    assert result["worker"] == "worker_dev"


class DummyResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self):
        import json

        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass
