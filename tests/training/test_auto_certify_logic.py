from core.agent_profile import AgentIdentity
from core.trust_evaluator import auto_certify


def test_auto_certify_grants_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = AgentIdentity(
        name="demo",
        role="critic",
        traits={},
        skills=[],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[],
        training_progress={"rev": "complete"},
        training_log=[],
    )
    profile.save()
    result = auto_certify("demo", "rev")
    assert result
    reloaded = AgentIdentity.load("demo")
    assert any(c.get("id") == "rev" for c in reloaded.certified_skills)
