from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, AgentRunContext


class DummyClient:
    def __init__(self, agents):
        self._agents = agents

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def get(self, url):
        return DummyResp({"agents": self._agents})

    def post(self, url, json=None, timeout=10):
        return DummyResp({})


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_agent_selection_lowest_cost(monkeypatch):
    agents = [
        {
            "id": "a1",
            "name": "a1",
            "url": "http://a1",
            "capabilities": ["demo"],
            "skills": ["demo"],
            "estimated_cost_per_token": 0.0005,
            "avg_response_time": 0.2,
            "load_factor": 0.2,
        },
        {
            "id": "a2",
            "name": "a2",
            "url": "http://a2",
            "capabilities": ["demo"],
            "skills": ["demo"],
            "estimated_cost_per_token": 0.0001,
            "avg_response_time": 0.3,
            "load_factor": 0.1,
        },
    ]

    monkeypatch.setattr("httpx.Client", lambda: DummyClient(agents))
    service = TaskDispatcherService()
    monkeypatch.setattr(
        service, "_run_agent", lambda a, c: AgentRunContext(agent_id=a["id"])
    )
    ctx = service.dispatch_task(TaskContext(task_type="demo"), mode="single")
    assert ctx.agent_selection == "a2"
