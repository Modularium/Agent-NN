from core.model_context import ModelContext, TaskContext, AccessText, AgentRunContext
from core.privacy import AccessLevel
from core.governance import AgentContract
from services.task_dispatcher.service import TaskDispatcherService


def test_dispatcher_applies_redaction(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    contract = AgentContract(
        agent="a1",
        allowed_roles=[],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={},
        max_access_level=AccessLevel.INTERNAL,
    )
    contract.save()

    service = TaskDispatcherService()
    agent = {"id": "a1", "name": "a1", "role": "demo", "url": "http://a1"}
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [])

    def fake_run(a, c):
        return AgentRunContext(agent_id=a["id"], result=c.task_context.input_data.text)

    monkeypatch.setattr(service, "_run_agent", fake_run)

    task = TaskContext(
        task_type="demo",
        input_data=AccessText(text="secret", access=AccessLevel.CONFIDENTIAL),
    )
    ctx = service.dispatch_task(task)
    assert ctx.agents[0].result == "[REDACTED]"
