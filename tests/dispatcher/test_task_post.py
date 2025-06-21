from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext


class DummyResponse:
    def json(self):
        return {"agents": [{"id": "a1", "capabilities": ["demo"]}]}

    def raise_for_status(self):
        pass


class DummyClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def get(self, url):
        return DummyResponse()


def test_dispatch_returns_context(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    service = TaskDispatcherService()
    ctx = service.dispatch_task(TaskContext(task_type="demo"))
    assert ctx.agent_selection == "a1"
    assert ctx.task_context.task_type == "demo"
