import threading

from core.agent_bus import reset
from core.model_context import ModelContext, TaskContext
from services.agent_worker.demo_agents.interactive_writer_agent import (
    InteractiveWriterAgent,
)
from services.agent_worker.demo_agents.critic_agent import CriticAgent


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
        return DummyResp({"completion": "draft", "tokens_used": 1})


def test_realtime_interaction(monkeypatch):
    reset("interactive_writer_agent")
    reset("critic_agent")
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())

    writer = InteractiveWriterAgent(llm_url="http://llm")
    critic = CriticAgent()
    ctx = ModelContext(task_context=TaskContext(task_type="demo", description="x"))

    def run_writer():
        nonlocal ctx
        ctx = writer.run(ctx)

    t = threading.Thread(target=run_writer)
    t.start()
    critic.process_bus()
    t.join()

    assert ctx.metrics.get("iterations") == 2
