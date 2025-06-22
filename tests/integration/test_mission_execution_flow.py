from core.missions import AgentMission
from core.agent_profile import AgentIdentity
from core.governance import AgentContract
from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, AgentRunContext


def test_mission_execution_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mission = AgentMission(
        id="m1",
        title="Demo",
        description="demo",
        steps=[
            {"task": "step1", "role": "writer", "skill_required": ["demo"], "deadline": None},
            {"task": "step2", "role": "writer", "skill_required": ["demo"], "deadline": None},
        ],
        rewards={"tokens": 10},
        team_mode="solo",
        mentor_required=False,
        track_id=None,
    )
    mission.save()
    profile = AgentIdentity(
        name="a1",
        role="writer",
        traits={},
        skills=["demo"],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[],
        training_progress={},
        training_log=[],
    )
    profile.save()
    contract = AgentContract(
        agent="a1",
        allowed_roles=["writer"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={},
    )
    contract.save()

    service = TaskDispatcherService()
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: [{
        "id": "a1", "name": "a1", "url": "http://a1", "capabilities": ["demo"], "role": "writer", "skills": ["demo"]
    }])
    monkeypatch.setattr(service, "_run_agent", lambda a, c: AgentRunContext(agent_id="a1"))

    service.dispatch_task(TaskContext(task_type="demo"), mission_id="m1", mission_step=0)
    service.dispatch_task(TaskContext(task_type="demo"), mission_id="m1", mission_step=1)
    contract = AgentContract.load("a1")
    assert contract.constraints.get("bonus_tokens") == 10
