from datetime import datetime

import importlib
from core.agent_profile import AgentIdentity


def test_aggregation_logic(tmp_path, monkeypatch):
    monkeypatch.setenv("RATING_DIR", str(tmp_path / "ratings"))
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path / "profiles"))
    rep = importlib.import_module("core.reputation")
    importlib.reload(rep)

    profile = AgentIdentity(
        name="a2",
        role="",
        traits={},
        skills=[],
        memory_index=None,
        created_at="now",
    )
    profile.save()

    rep.save_rating(
        rep.AgentRating("a1", "a2", None, 0.8, None, [], datetime.utcnow().isoformat())
    )
    rep.save_rating(
        rep.AgentRating("a3", "a2", None, 0.4, None, [], datetime.utcnow().isoformat())
    )
    score = rep.update_reputation("a2")
    assert abs(score - 0.6) < 0.001
    prof = AgentIdentity.load("a2")
    assert prof.reputation_score == score
    assert round(rep.aggregate_score("a2"), 3) == round(score, 3)
