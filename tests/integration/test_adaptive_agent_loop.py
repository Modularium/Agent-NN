import importlib
from core import agent_evolution
from services.agent_worker.demo_agents.writer_agent import WriterAgent
from core.agent_profile import AgentIdentity
from core.model_context import ModelContext, TaskContext


def test_adaptive_agent_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_EVOLVE", "true")
    monkeypatch.setenv("AGENT_EVOLVE_INTERVAL", "2")
    called = {}

    def fake_evolve(profile, history, mode="heuristic"):
        called["done"] = True
        profile.traits["evolved"] = True
        return profile

    monkeypatch.setattr(agent_evolution, "evolve_profile", fake_evolve)
    importlib.reload(agent_evolution)
    agent = WriterAgent(llm_url="http://llm")
    ctx = ModelContext(task_context=TaskContext(task_type="demo", description="t"))
    agent.run(ctx)
    agent.run(ctx)
    profile = AgentIdentity.load("writer_agent")
    assert called.get("done")
    assert profile.traits["evolved"] is True
