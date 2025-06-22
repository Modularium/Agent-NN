from core.teams import AgentTeam
from core.agent_profile import AgentIdentity


def test_team_membership(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    team = AgentTeam(
        id="t1",
        name="demo",
        members=[],
        shared_goal=None,
        skills_focus=[],
        coordinator="alice",
        created_at="2024-01-01T00:00:00Z",
    )
    team.save()
    profile = AgentIdentity(
        name="bob",
        role="",
        traits={},
        skills=[],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[],
        training_progress={},
        training_log=[],
    )
    profile.team_id = "t1"
    profile.team_role = "apprentice"
    profile.save()
    team.members.append("bob")
    team.save()
    reloaded = AgentIdentity.load("bob")
    assert reloaded.team_id == "t1"
    loaded = AgentTeam.load("t1")
    assert "bob" in loaded.members


