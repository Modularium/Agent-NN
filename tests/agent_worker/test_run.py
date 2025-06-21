from services.agent_worker.sample_agent.service import SampleAgentService
from core.model_context import ModelContext, TaskContext


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json, timeout):
        return DummyResponse(self.payload)


def test_run_returns_updated_context(monkeypatch):
    payload = {"completion": "hi", "tokens_used": 2, "provider": "dummy"}
    monkeypatch.setattr("httpx.Client", lambda: DummyClient(payload))
    service = SampleAgentService(llm_url="http://llm")
    ctx = ModelContext(task_context=TaskContext(task_type="demo"))
    out = service.run(ctx)
    assert out.result == "hi"
    assert out.metrics["tokens_used"] == 2
