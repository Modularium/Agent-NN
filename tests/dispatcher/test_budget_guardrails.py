from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, AgentRunContext


def test_budget_exceeded_prevents_run(monkeypatch):
    service = TaskDispatcherService()
    agent = {"id": "a1", "url": "http://worker", "capabilities": ["demo"]}
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr(
        service, "_fetch_history", lambda sid: [{"metrics": {"tokens_used": 1}}]
    )
    called = {"run": False}

    def fake_run(a, ctx):
        called["run"] = True
        return AgentRunContext(
            agent_id=a["id"], metrics={"tokens_used": 1}, result="ok"
        )

    monkeypatch.setattr(service, "_run_agent", fake_run)
    ctx = service.dispatch_task(
        TaskContext(task_type="demo"), session_id="s1", max_tokens=1, task_value=0.5
    )
    assert ctx.warning == "budget exceeded"
    assert not called["run"]


def test_budget_updated_after_run(monkeypatch):
    service = TaskDispatcherService()
    agent = {"id": "a1", "url": "http://worker", "capabilities": ["demo"]}
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr(
        service, "_fetch_history", lambda sid: [{"metrics": {"tokens_used": 2}}]
    )

    def fake_run(a, ctx):
        return AgentRunContext(
            agent_id=a["id"], metrics={"tokens_used": 3}, result="ok"
        )

    monkeypatch.setattr(service, "_run_agent", fake_run)
    ctx = service.dispatch_task(
        TaskContext(task_type="demo"), session_id="s1", max_tokens=4
    )
    assert ctx.token_spent == 5
    assert ctx.warning == "budget exceeded"
