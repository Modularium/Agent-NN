import pytest

from core.governance import AgentContract
from core.model_context import AgentRunContext, TaskContext
from services.task_dispatcher.service import TaskDispatcherService


@pytest.mark.integration
def test_trust_based_role_change(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="a1",
        allowed_roles=["retriever"],
        temp_roles=["analyst"],
        max_tokens=100,
        trust_level_required=0.0,
        constraints={},
    ).save()

    service = TaskDispatcherService()
    agent = {"id": "a1", "name": "a1", "role": "analyst", "url": "http://a1"}
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [])
    monkeypatch.setattr(
        service,
        "_run_agent",
        lambda a, c: AgentRunContext(agent_id=a["id"], result="ok"),
    )

    ctx = service.dispatch_task(TaskContext(task_type="demo"))
    assert ctx.result == "ok"
    assert "analyst" in ctx.elevated_roles
    contract = AgentContract.load("a1")
    assert not contract.temp_roles
