from services.agent_registry.schemas import AgentInfo


def test_agentinfo_cost_fields() -> None:
    info = AgentInfo(name="a", url="http://a")
    assert hasattr(info, "estimated_cost_per_token")
    assert hasattr(info, "avg_response_time")
    assert hasattr(info, "load_factor")
