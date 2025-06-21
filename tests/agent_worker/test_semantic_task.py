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
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json, timeout=10):
        if url.endswith("/vector_search"):
            return DummyResp({"matches": [{"id": "1", "text": "doc", "distance": 0.1}], "model": "dummy"})
        if url.endswith("/generate"):
            return DummyResp({"completion": "ans", "tokens_used": 5, "provider": "dummy"})
        raise AssertionError("unexpected url" + url)


def test_semantic_task(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    service = SampleAgentService(llm_url="http://llm", vector_url="http://vec")
    ctx = ModelContext(task_context=TaskContext(task_type="semantic", description="q"))
    out = service.run(ctx)
    assert out.result["generated_response"] == "ans"
    assert out.result["sources"][0]["text"] == "doc"
    assert out.metrics["tokens_used"] == 5
