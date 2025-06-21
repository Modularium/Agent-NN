from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return {
            **self._payload,
            "agent_selection": "a1",
            "result": {
                "generated_response": "ok",
                "sources": [],
                "embedding_distance_avg": 0.0,
            },
            "metrics": {"tokens_used": 1},
        }

    def raise_for_status(self):
        pass


class DummyClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json, timeout=10):
        return DummyResp(json)


def test_semantic_chain(monkeypatch):
    dispatcher = TaskDispatcherService()
    agent = {"id": "a1", "url": "http://worker", "capabilities": ["semantic"]}
    monkeypatch.setattr(dispatcher, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    task = TaskContext(task_type="semantic")
    ctx = dispatcher.dispatch_task(task)
    assert ctx.result["generated_response"] == "ok"
