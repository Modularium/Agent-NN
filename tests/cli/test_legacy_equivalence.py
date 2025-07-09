import pytest
pytest.importorskip("pydantic")
pytestmark = pytest.mark.heavy
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

sys.modules.setdefault(
    "agentnn.session.session_manager",
    types.SimpleNamespace(SessionManager=object),
)
sys.modules.setdefault(
    "agentnn.mcp.mcp_ws", types.SimpleNamespace(ws_server=types.SimpleNamespace(broadcast=lambda *a, **k: None))
)
sys.modules.setdefault(
    "agentnn.mcp.mcp_server", types.SimpleNamespace(create_app=lambda: None)
)
import sdk.nn_models as _nn
sys.modules.setdefault("sdk.cli.nn_models", _nn)
DummySettings = type(
    "DummySettings",
    (),
    {"load": classmethod(lambda cls: cls()), "__init__": lambda self: None},
)
sys.modules.setdefault("sdk.cli.config", types.SimpleNamespace(SDKSettings=DummySettings))
sys.modules.setdefault(
    "core.config", types.SimpleNamespace(settings=types.SimpleNamespace(model_dump=lambda: {}))
)

from sdk.cli.main import app  # noqa: E402


def test_agent_register_equiv(monkeypatch, tmp_path):
    """New agent register behaves like legacy deploy."""
    from sdk.cli.commands import agent as agent_cmd
    from sdk.cli.commands import agentctl

    cfg = tmp_path / "agent.yaml"
    cfg.write_text("id: demo")

    monkeypatch.setattr(agent_cmd, "load_agent_file", lambda p: {"id": "demo"})
    monkeypatch.setattr(agentctl, "load_agent_file", lambda p: {"id": "demo"})

    called = {}

    class DummyRegistry:
        def __init__(self, endpoint: str) -> None:
            called["endpoint"] = endpoint

        def deploy(self, data):
            called["data"] = data
            return {"ok": True}

    monkeypatch.setattr(agent_cmd, "AgentRegistry", DummyRegistry)
    monkeypatch.setattr(agentctl, "AgentRegistry", DummyRegistry)

    runner = CliRunner()
    result_new = runner.invoke(app, ["agent", "register", str(cfg)])
    result_old = runner.invoke(app, ["agent", "deploy", str(cfg)])

    assert result_new.exit_code == 0 and result_old.exit_code == 0
    assert called.get("data") == {"id": "demo"}
