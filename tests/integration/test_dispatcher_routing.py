from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, AccessText, AgentRunContext


def test_legacy_task_mapping(monkeypatch):
    svc = TaskDispatcherService(
        registry_url="",
        session_url="",
        coordinator_url="",
        coalition_url="",
        routing_url="",
    )

    monkeypatch.setattr(
        svc, "_fetch_agents", lambda t: [{"id": "1", "name": "dev", "role": "dev"}]
    )
    monkeypatch.setattr(svc, "_governance_allowed", lambda a, c: True)
    monkeypatch.setattr(svc, "_endorsement_allowed", lambda a, c: True)
    monkeypatch.setattr(svc, "_skills_allowed", lambda a, c: True)

    monkeypatch.setattr(
        svc,
        "_run_agent",
        lambda a, c: AgentRunContext(agent_id=a["id"], role=a["role"], result="ok"),
    )

    task = TaskContext(task_type="say_hello", input_data=AccessText(text="hi"))
    ctx = svc.dispatch_task(task)
    assert ctx.task_context.task_type == "dev"
