from core.trust_evaluator import calculate_trust


def test_calculate_trust_basic():
    context = [
        {
            "agent_id": "a1",
            "success": True,
            "feedback_score": 0.8,
            "metrics": {"tokens_used": 100},
            "expected_tokens": 120,
            "error": None,
        },
        {
            "agent_id": "a1",
            "success": False,
            "feedback_score": 0.6,
            "metrics": {"tokens_used": 50},
            "expected_tokens": 50,
            "error": "oops",
        },
    ]
    score = calculate_trust("a1", context)
    assert 0.0 <= score <= 1.0
