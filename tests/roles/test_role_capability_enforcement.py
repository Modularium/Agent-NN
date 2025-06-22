from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, AgentRunContext
from core.governance import AgentContract


def test_role_capabilities_trim(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="a1",
        allowed_roles=["writer"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={},
    ).save()
    service = TaskDispatcherService()
    agent = {"id": "a1", "name": "a1", "role": "writer", "url": "http://a1"}
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    memory = [{"output": i} for i in range(10)]
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [{"memory": memory}])
    monkeypatch.setattr(
        service,
        "_run_agent",
        lambda a, c: AgentRunContext(agent_id=a["id"], metrics={"tokens_used": 1}, result="ok"),
    )
    ctx = service.dispatch_task(TaskContext(task_type="demo"), session_id="s1", max_tokens=5000)
    assert ctx.max_tokens == 2000
    assert ctx.applied_limits["max_context_size"] == 5
    assert ctx.memory is not None and len(ctx.memory) == 5
