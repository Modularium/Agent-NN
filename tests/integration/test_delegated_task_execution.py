import os
from core.agent_profile import AgentIdentity
from core.governance import AgentContract
from core.model_context import TaskContext, AgentRunContext
from services.task_dispatcher.service import TaskDispatcherService
from core.delegation import grant_delegation


def test_delegated_task_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.environ["AGENT_PROFILE_DIR"] = str(tmp_path / "profiles")
    os.environ["DELEGATION_DIR"] = str(tmp_path / "delegations")

    AgentIdentity(name="coord", role="coordinator", traits={}, skills=[], memory_index=None, created_at="now").save()
    AgentIdentity(name="rev", role="reviewer", traits={}, skills=[], memory_index=None, created_at="now").save()
    AgentContract(agent="rev", allowed_roles=["reviewer"], max_tokens=0, trust_level_required=0.0, constraints={}).save()

    grant_delegation("coord", "rev", "reviewer", "task")

    service = TaskDispatcherService()
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [{
        "id": "rev", "name": "rev", "url": "http://rev", "capabilities": ["demo"], "role": "reviewer"
    }])
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [])
    monkeypatch.setattr(service, "_run_agent", lambda a, ctx: AgentRunContext(agent_id="rev"))

    ctx = service.dispatch_task(TaskContext(task_type="demo"))
    assert ctx.delegate_info and ctx.delegate_info.get("delegator") == "coord"
