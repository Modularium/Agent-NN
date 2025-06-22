from core.agent_profile import AgentIdentity
from core.agent_evolution import evolve_profile


def test_heuristic_evolution():
    profile = AgentIdentity(
        name="test",
        role="",
        traits={},
        skills=[],
        memory_index=None,
        created_at="now",
    )
    history = [{"rating": "good"}, {"rating": "bad"}, {"rating": "good"}]
    updated = evolve_profile(profile, history, mode="heuristic")
    assert 0.0 <= updated.traits["precision"] <= 1.0
    assert 0.0 <= updated.traits["harshness"] <= 1.0


def test_llm_evolution(monkeypatch):
    class DummyResp:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {"completion": '{"traits": {"creativity": 1}, "skills": ["a"]}'}

        def raise_for_status(self):
            pass

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json, timeout=15):
            return DummyResp()

    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    profile = AgentIdentity(
        name="test",
        role="",
        traits={},
        skills=[],
        memory_index=None,
        created_at="now",
    )
    updated = evolve_profile(profile, [], mode="llm")
    assert updated.traits["creativity"] == 1
    assert "a" in updated.skills
