from core.model_context import TaskContext, AgentRunContext
from services.task_dispatcher.service import TaskDispatcherService
from core.governance import AgentContract
from core.audit_log import AuditLog
from core.privacy import AccessLevel


def test_violation_logged(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    contract = AgentContract(
        agent="a1",
        allowed_roles=["demo"],
        max_tokens=100,
        trust_level_required=0.0,
        constraints={},
        max_access_level=AccessLevel.INTERNAL,
    )
    contract.save()

    service = TaskDispatcherService()
    service.audit = AuditLog(log_dir=tmp_path)
    agent = {"id": "a1", "name": "a1", "role": "other", "url": "http://a1"}
    monkeypatch.setattr(service, "_fetch_agents", lambda c: [agent])
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [])
    monkeypatch.setattr(
        service, "_run_agent", lambda a, c: AgentRunContext(agent_id=a["id"])
    )

    ctx = service.dispatch_task(TaskContext(task_type="demo"))
    entries = service.audit.by_context(ctx.uuid)
    assert any(e["action"] == "role_rejected" for e in entries)
