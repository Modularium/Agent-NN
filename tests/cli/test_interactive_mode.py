import pytest
pytest.importorskip("pydantic")
pytestmark = pytest.mark.heavy
from typer.testing import CliRunner

from sdk.cli.main import app


def test_agent_register_wizard(monkeypatch):
    from sdk.cli.commands import agent as agent_cmd

    class DummyReg:
        def __init__(self, endpoint: str) -> None:
            pass

        def deploy(self, data):
            return {"ok": data["id"]}

    monkeypatch.setattr(agent_cmd, "AgentRegistry", DummyReg)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["agent", "register", "--interactive"],
        input="demo\nworker\n\nDemo agent\n",
    )
    assert result.exit_code == 0
    assert "demo" in result.stdout


def test_agent_list_markdown(monkeypatch):
    from sdk.cli.commands import agent as agent_cmd

    monkeypatch.setattr(
        agent_cmd.AgentClient,
        "list_agents",
        lambda self: {"agents": [{"name": "a", "role": "r"}]},
    )
    runner = CliRunner()
    result = runner.invoke(app, ["agent", "list", "--output", "json"])
    assert result.exit_code == 0
    assert '"name": "a"' in result.stdout
