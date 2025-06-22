from core.agent_profile import AgentIdentity
from core.governance import AgentContract
from core.level_evaluator import check_level_up
from core.levels import AgentLevel, save_level


def test_level_progression(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path / "contracts"))
    from core import levels

    monkeypatch.setattr(levels, "LEVEL_DIR", tmp_path / "levels")

    save_level(
        AgentLevel(
            id="basic",
            title="Basic",
            trust_required=0.5,
            skills_required=["demo"],
            unlocks={"roles": ["reviewer"]},
        )
    )

    profile = AgentIdentity(
        name="tester",
        role="writer",
        traits={},
        skills=[],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[{"id": "demo", "granted_at": "now", "expires_at": None}],
    )
    profile.save()
    history = [
        {
            "agent_id": "tester",
            "success": True,
            "feedback_score": 1.0,
            "metrics": {"tokens_used": 10},
            "expected_tokens": 10,
            "error": None,
        }
        for _ in range(5)
    ]
    AgentContract(
        agent="tester",
        allowed_roles=["writer"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={"task_history": history},
    ).save()

    new_level = check_level_up(profile)
    assert new_level == "basic"
    updated = AgentIdentity.load("tester")
    assert updated.current_level == "basic"
