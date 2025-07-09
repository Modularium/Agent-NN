import pytest
pytest.importorskip("pydantic")
pytestmark = pytest.mark.heavy
import sys
import types
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "mcp",
    types.SimpleNamespace(
        types=types.SimpleNamespace(BaseModel=object, Field=lambda *a, **k: None)
    ),
)

sys.modules.setdefault(
    "agentnn.session.session_manager",
    types.SimpleNamespace(SessionManager=object),
)
sys.modules.setdefault(
    "agentnn.mcp.mcp_ws",
    types.SimpleNamespace(
        ws_server=types.SimpleNamespace(broadcast=lambda *a, **k: None)
    ),
)
sys.modules.setdefault(
    "agentnn.mcp.mcp_server", types.SimpleNamespace(create_app=lambda: None)
)
sys.modules.setdefault(
    "mlflow",
    types.SimpleNamespace(
        start_run=lambda *a, **k: None,
        set_tag=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        set_tracking_uri=lambda *a, **k: None,
        tracking=types.SimpleNamespace(
            MlflowClient=lambda: types.SimpleNamespace(
                list_experiments=lambda: [],
                get_run=lambda run_id: types.SimpleNamespace(
                    info=types.SimpleNamespace(run_id=run_id, status="FINISHED"),
                    data=types.SimpleNamespace(metrics={}, params={}),
                ),
            )
        ),
    ),
)
import sdk.nn_models as _nn

sys.modules.setdefault("sdk.cli.nn_models", _nn)
DummySettings = type(
    "DummySettings",
    (),
    {"load": classmethod(lambda cls: cls()), "__init__": lambda self: None},
)
sys.modules.setdefault(
    "sdk.cli.config", types.SimpleNamespace(SDKSettings=DummySettings)
)
sys.modules.setdefault(
    "core.config",
    types.SimpleNamespace(settings=types.SimpleNamespace(model_dump=lambda: {})),
)

from sdk.cli.main import app  # noqa: E402


def test_agentnn_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "session" in result.stdout


def test_session_start(monkeypatch, tmp_path):
    from sdk.cli.commands import session as session_cmd

    monkeypatch.setattr(session_cmd.manager, "create_session", lambda: "sid")
    monkeypatch.setattr(session_cmd.manager, "add_agent", lambda *a, **k: None)
    monkeypatch.setattr(session_cmd.manager, "run_task", lambda *a, **k: None)

    tpl = tmp_path / "tpl.yaml"
    tpl.write_text("agents: []\ntasks: []")
    runner = CliRunner()
    result = runner.invoke(app, ["session", "start", str(tpl)])
    assert result.exit_code == 0
    assert "sid" in result.stdout


def test_session_snapshot(monkeypatch):
    from sdk.cli.commands import session as session_cmd

    monkeypatch.setattr(session_cmd.manager, "get_session", lambda sid: {})
    monkeypatch.setattr(
        session_cmd.snapshot_store, "save_snapshot", lambda sid, data: "snap"
    )
    runner = CliRunner()
    result = runner.invoke(app, ["session", "snapshot", "sid"])
    assert result.exit_code == 0
    assert "snap" in result.stdout


def test_agent_deploy(monkeypatch, tmp_path):
    from sdk.cli.commands import agentctl

    monkeypatch.setattr(agentctl, "load_agent_file", lambda p: {"id": "demo"})

    class DummyRegistry:
        def __init__(self, endpoint: str) -> None:
            pass

        def deploy(self, data):
            return {"ok": True}

    monkeypatch.setattr(agentctl, "AgentRegistry", DummyRegistry)
    cfg = tmp_path / "agent.yaml"
    cfg.write_text("id: demo")
    runner = CliRunner()
    result = runner.invoke(app, ["agent", "deploy", str(cfg)])
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_dispatch(monkeypatch):
    from sdk.cli.commands import dispatch as dispatch_mod

    class DummyClient:
        def dispatch_task(self, ctx):
            return {"ok": True}

    monkeypatch.setattr(dispatch_mod, "AgentClient", lambda: DummyClient())
    runner = CliRunner()
    result = runner.invoke(app, ["task", "dispatch", "hello"])
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_dispatch_with_tool(monkeypatch):
    from sdk.cli.commands import dispatch as dispatch_mod

    class DummyClient:
        def dispatch_task(self, ctx):
            assert ctx.agent_selection == "dynamic_architecture"
            return {"ok": True}

    monkeypatch.setattr(dispatch_mod, "AgentClient", lambda: DummyClient())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["task", "dispatch", "hello", "--tool", "dynamic_architecture"],
    )
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_session_list(monkeypatch):
    from sdk.cli.commands import session as session_cmd

    monkeypatch.setattr(
        session_cmd.AgentClient, "list_sessions", lambda self: {"sessions": []}
    )
    runner = CliRunner()
    result = runner.invoke(app, ["session", "list"])
    assert result.exit_code == 0
    assert "sessions" in result.stdout


def test_agent_register(monkeypatch, tmp_path):
    from sdk.cli.commands import agent as agent_cmd

    cfg = tmp_path / "agent.yaml"
    cfg.write_text("id: demo")

    monkeypatch.setattr(agent_cmd, "load_agent_file", lambda p: {"id": "demo"})

    class DummyRegistry:
        def __init__(self, endpoint: str) -> None:
            pass

        def deploy(self, data):
            return {"ok": True}

    monkeypatch.setattr(agent_cmd, "AgentRegistry", DummyRegistry)
    runner = CliRunner()
    result = runner.invoke(app, ["agent", "register", str(cfg)])
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_prompt_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["prompt", "refine", "bad  prompt"])
    assert result.exit_code == 0
    assert "bad prompt" in result.stdout
    result = runner.invoke(app, ["prompt", "quality", "good prompt text"])
    assert result.exit_code == 0
    assert "0." in result.stdout or "1" in result.stdout
