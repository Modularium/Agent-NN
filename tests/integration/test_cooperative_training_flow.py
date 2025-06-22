from core.agent_profile import AgentIdentity
from core.team_knowledge import broadcast_insight
from core.training import TrainingPath, save_training_path
from core.teams import AgentTeam
from core.audit_log import AuditLog


def test_cooperative_training_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tp = TrainingPath(
        id="rev",
        target_skill="rev",
        prerequisites=[],
        method="prompt",
        evaluation_prompt="test",
        certifier_agent="mentor",
        mentor_required=False,
        min_trust=0.0,
        team_mode="cooperative",
    )
    save_training_path(tp)
    team = AgentTeam(
        id="t1",
        name="demo",
        members=["a1", "a2"],
        shared_goal=None,
        skills_focus=["rev"],
        coordinator="a1",
        created_at="2024-01-01T00:00:00Z",
    )
    team.save()
    for agent in ["a1", "a2"]:
        profile = AgentIdentity(
            name=agent,
            role="",
            traits={},
            skills=[],
            memory_index=None,
            created_at="2024-01-01T00:00:00Z",
            certified_skills=[],
            training_progress={"rev": "complete"},
            training_log=[],
        )
        profile.team_id = "t1"
        profile.save()
    broadcast_insight("a1", "rev", {"done": True})
    broadcast_insight("a2", "rev", {"done": True})
    log = AuditLog(log_dir="audit")
    entries = log.by_context("t1")
    assert any(e.get("action") == "cooperative_training_success" for e in entries)

