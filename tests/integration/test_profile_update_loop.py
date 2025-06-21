import importlib
import os
from services.agent_worker.demo_agents.critic_agent import CriticAgent
from core import agent_profile



def test_profile_update_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path))
    importlib.reload(agent_profile)
    agent = CriticAgent()
    agent.vote("some short text")
    profile = agent_profile.AgentIdentity.load("critic_agent")
    assert profile.traits.get("ratings") == 1
