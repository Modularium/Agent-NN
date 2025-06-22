from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, AgentRunContext


def test_budget_accumulates(monkeypatch):
    service = TaskDispatcherService()
    agent = {"id": "a1", "url": "http://worker", "capabilities": ["demo"]}
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [])

    def fake_run(a, ctx):
        return AgentRunContext(
            agent_id=a["id"], metrics={"tokens_used": 1}, result="ok"
        )

    monkeypatch.setattr(service, "_run_agent", fake_run)
    ctx1 = service.dispatch_task(
        TaskContext(task_type="demo"), session_id="s1", max_tokens=2
    )
    assert ctx1.token_spent == 1
    monkeypatch.setattr(
        service,
        "_fetch_history",
        lambda sid: [{"metrics": {"tokens_used": ctx1.token_spent}}],
    )
    ctx2 = service.dispatch_task(
        TaskContext(task_type="demo"), session_id="s1", max_tokens=2
    )
    assert ctx2.token_spent == 2
    assert ctx2.warning is None
    monkeypatch.setattr(
        service,
        "_fetch_history",
        lambda sid: [{"metrics": {"tokens_used": ctx2.token_spent}}],
    )
    ctx3 = service.dispatch_task(
        TaskContext(task_type="demo"), session_id="s1", max_tokens=2
    )
    assert ctx3.warning == "budget exceeded"
