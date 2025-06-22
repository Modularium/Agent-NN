from services.task_dispatcher.service import TaskDispatcherService
from core.agent_profile import AgentIdentity
from core.model_context import TaskContext, AgentRunContext


def test_missing_skill_sets_warning(monkeypatch):
    service = TaskDispatcherService()
    agent = {
        "id": "a1",
        "name": "a1",
        "url": "http://a1",
        "capabilities": ["demo"],
        "role": "writer",
    }
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [agent])
    monkeypatch.setattr(service, "_fetch_history", lambda sid: [])
    monkeypatch.setattr(
        service, "_run_agent", lambda a, ctx: AgentRunContext(agent_id=a["id"])
    )

    def fake_load(name: str) -> AgentIdentity:
        return AgentIdentity(
            name=name,
            role="writer",
            traits={},
            skills=[],
            memory_index=None,
            created_at="2024-01-01T00:00:00Z",
            certified_skills=[],
        )

    monkeypatch.setattr(AgentIdentity, "load", staticmethod(fake_load))
    ctx = service.dispatch_task(
        TaskContext(task_type="demo"),
        required_skills=["demo"],
        enforce_certification=True,
    )
    assert ctx.warning == "missing_skills"
