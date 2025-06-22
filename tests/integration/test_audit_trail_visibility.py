from core.model_context import TaskContext, AgentRunContext
from services.task_dispatcher.service import TaskDispatcherService
from core.audit_log import AuditLog


def test_audit_trail(monkeypatch, tmp_path):
    service = TaskDispatcherService()
    service.audit = AuditLog(log_dir=tmp_path)

    monkeypatch.setattr(
        service,
        "_fetch_agents",
        lambda c: [{"id": "a1", "name": "a1", "url": "http://a1"}],
    )
    monkeypatch.setattr(
        service, "_run_agent", lambda a, c: AgentRunContext(agent_id=a["id"])
    )

    ctx = service.dispatch_task(TaskContext(task_type="demo"))
    assert ctx.audit_trace
