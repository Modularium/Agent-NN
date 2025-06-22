from core.governance import AgentContract
from core.rewards import grant_rewards


def test_reward_assignment(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="a1",
        allowed_roles=["writer"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={},
    ).save()

    grant_rewards("a1", {"tokens": 100, "roles": ["critic"]})
    c = AgentContract.load("a1")
    assert c.constraints.get("bonus_tokens") == 100
    assert "critic" in c.allowed_roles
