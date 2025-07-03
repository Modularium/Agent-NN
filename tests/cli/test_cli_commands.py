from typer.testing import CliRunner

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "mcp",
    types.SimpleNamespace(
        types=types.SimpleNamespace(BaseModel=object, Field=lambda *a, **k: None)
    ),
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

from sdk.cli.main import app


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
