from datetime import datetime

from core.agent_profile import AgentIdentity
from core.trust_evaluator import auto_certify


def test_skill_evaluation_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = AgentIdentity(
        name="demo",
        role="critic",
        traits={},
        skills=[],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[],
        training_progress={},
        training_log=[],
    )
    profile.save()
    profile.training_log.append(
        {
            "skill": "rev",
            "evaluation_score": 0.9,
            "last_attempted": datetime.utcnow().isoformat(),
        }
    )
    profile.training_progress["rev"] = "complete"
    profile.save()
    assert auto_certify("demo", "rev")
