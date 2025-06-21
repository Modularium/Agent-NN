from services.agent_worker.demo_agents.writer_agent import WriterAgent
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
        self.prompt = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json, timeout=10):
        if url.endswith("/generate"):
            self.prompt = json["prompt"]
            return DummyResp({"completion": "ok", "tokens_used": 1})
        raise AssertionError("unexpected url" + url)


def test_writer_receives_memory(monkeypatch):
    client = DummyClient()
    monkeypatch.setattr("httpx.Client", lambda: client)
    agent = WriterAgent(llm_url="http://llm")
    ctx = ModelContext(
        task_context=TaskContext(task_type="demo", description="new"),
        memory=[{"output": "old"}],
    )
    agent.run(ctx)
    assert "old" in client.prompt
