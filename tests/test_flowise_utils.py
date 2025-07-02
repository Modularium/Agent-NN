import json
from utils.flowise import agent_config_to_flowise


def test_agent_config_to_flowise():
    cfg = {
        "name": "demo_agent",
        "domain": "demo",
        "capabilities": ["a"],
        "tools": ["t1"],
        "model_config": {"model": "gpt"},
        "created_at": "2024-01-01",
        "version": "1.0.0",
    }
    result = agent_config_to_flowise(cfg)
    assert result["id"] == "demo_agent"
    assert result["tools"] == ["t1"]
    assert result["llm"]["model"] == "gpt"
