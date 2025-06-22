from datetime import datetime
import importlib

from core.agent_profile import AgentIdentity


def test_endorsement_logic(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOMMEND_DIR", str(tmp_path / "recs"))
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path / "profiles"))
    mod = importlib.import_module("core.trust_network")
    importlib.reload(mod)

    profile = AgentIdentity(
        name="a2",
        role="",
        traits={},
        skills=[],
        memory_index=None,
        created_at="now",
    )
    profile.save()

    rec = mod.AgentRecommendation(
        from_agent="a1",
        to_agent="a2",
        role="reviewer",
        confidence=0.9,
        comment="great",
        created_at=datetime.utcnow().isoformat(),
    )
    mod.record_recommendation(rec)
    recs = mod.load_recommendations("a2")
    assert recs and recs[0].role == "reviewer"
    prof = AgentIdentity.load("a2")
    assert "a1" in prof.trusted_by
