from core.trust_evaluator import update_trust_usage
from core.governance import AgentContract


def test_trust_score_updates(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="agent1",
        allowed_roles=["writer"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={"trust_score": 0.5},
    ).save()
    update_trust_usage("agent1", 1500, 2000)
    c = AgentContract.load("agent1")
    assert c.constraints["trust_score"] > 0.5
    update_trust_usage("agent1", 2500, 2000)
    c2 = AgentContract.load("agent1")
    assert c2.constraints["trust_score"] < c.constraints["trust_score"]
