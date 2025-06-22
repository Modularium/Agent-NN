from core.governance import AgentContract
from core.trust_evaluator import eligible_for_role


def test_dynamic_eligibility(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="agent1",
        allowed_roles=["retriever"],
        temp_roles=None,
        max_tokens=0,
        trust_level_required=0.5,
        constraints={
            "task_history": [{"agent_id": "agent1", "success": True} for _ in range(6)],
            "standing": 1.0,
        },
    ).save()

    assert eligible_for_role("agent1", "analyst")
