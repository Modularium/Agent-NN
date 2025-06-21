from services.task_dispatcher.service import TaskDispatcherService
from services.task_dispatcher.schemas import TaskRequest


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def get(self, url):
        if "context" in url:
            return DummyResp({"context": []})
        return DummyResp({"agents": [{"id": "a1", "url": "http://worker", "capabilities": ["demo"]}]})

    def post(self, url, json, timeout=10):
        return DummyResp(json | {"result": "ok"})


def test_dispatch_with_session(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    service = TaskDispatcherService(registry_url="http://reg", session_url="http://sess")
    req = TaskRequest(task_type="demo", session_id="s1")
    ctx = service.dispatch_task(req, session_id=req.session_id)
    assert ctx.session_id == "s1"
    assert ctx.agent_selection == "a1"
