import pytest
pytest.importorskip("pydantic")
from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
from typer.testing import CliRunner
from sdk.cli.main import app
from core.agent_profile import AgentIdentity
import core.agent_evolution as agent_evolution


def test_agent_evolve_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path))
    profile = AgentIdentity(name="demo", role="", traits={}, skills=[], memory_index=None, created_at="now")
    profile.save()

    def fake_evolve(agent, history, mode="llm"):
        agent.traits["x"] = 1
        return agent

    monkeypatch.setattr(agent_evolution, "evolve_profile", fake_evolve)
    runner = CliRunner()
    result = runner.invoke(app, ["agent", "evolve", "demo", "--mode", "heuristic"])
    assert result.exit_code == 0
    updated = AgentIdentity.load("demo")
    assert updated.traits["x"] == 1
