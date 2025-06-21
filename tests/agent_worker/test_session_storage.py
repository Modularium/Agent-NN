from services.agent_worker.sample_agent.service import SampleAgentService
from core.model_context import ModelContext, TaskContext


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def __init__(self):
        self.updated = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json, timeout=10):
        if url.endswith("/vector_search"):
            return DummyResp({"matches": [], "model": "dummy"})
        if url.endswith("/generate"):
            return DummyResp({"completion": "hi", "tokens_used": 1, "provider": "dummy"})
        if url.endswith("/update_context"):
            self.updated = True
            return DummyResp({"status": "ok"})
        raise AssertionError("unexpected url" + url)


def test_session_storage(monkeypatch):
    client = DummyClient()
    monkeypatch.setattr("httpx.Client", lambda: client)
    service = SampleAgentService(llm_url="http://llm", vector_url="http://vec", session_url="http://sess")
    ctx = ModelContext(task_context=TaskContext(task_type="demo"), session_id="s1")
    service.run(ctx)
    assert client.updated
