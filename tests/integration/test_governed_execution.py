import pytest

from core.model_context import TaskContext, AgentRunContext
from core.governance import AgentContract
from services.task_dispatcher.service import TaskDispatcherService


@pytest.mark.integration
def test_governed_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    contract = AgentContract(
        agent="a1",
        allowed_roles=["demo"],
        max_tokens=100,
        trust_level_required=0.4,
        constraints={},
    )
    contract.save()

    service = TaskDispatcherService()
    agent = {"id": "a1", "name": "a1", "role": "demo", "url": "http://a1"}
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [])
    monkeypatch.setattr(
        service, "_run_agent", lambda a, c: AgentRunContext(agent_id=a["id"], result="ok")
    )

    ctx = service.dispatch_task(TaskContext(task_type="demo"))
    assert ctx.result == "ok"
    assert ctx.warning is None
