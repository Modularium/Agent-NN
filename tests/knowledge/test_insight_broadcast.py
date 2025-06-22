from core.agent_profile import AgentIdentity
from core.team_knowledge import broadcast_insight, share_training_material
from core.teams import AgentTeam


def test_insight_broadcast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    team = AgentTeam(
        id="team1",
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
            training_progress={},
            training_log=[],
        )
        profile.team_id = "team1"
        profile.save()
    broadcast_insight("a1", "rev", {"info": 1})
    data = share_training_material("team1", "rev")
    assert data and data[0]["agent"] == "a1"

