import pytest
pytest.importorskip("pydantic")
from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
from typer.testing import CliRunner

from sdk.cli.main import app
from tools import registry as registry_mod


def test_cli_tools_list_builtin(monkeypatch):
    from sdk.cli.commands import tools as tools_cmd

    class DummyMgr:
        def list_plugins(self):
            return []

        def get(self, name):
            return None

    monkeypatch.setattr(tools_cmd, "PluginManager", lambda: DummyMgr())
    monkeypatch.setattr(
        registry_mod.ToolRegistry,
        "list_tools",
        classmethod(lambda cls: ["agent_nn_v2", "dynamic_architecture"]),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["tools", "list"])
    assert result.exit_code == 0
    assert "agent_nn_v2" in result.stdout
    assert "dynamic_architecture" in result.stdout
