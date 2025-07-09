import pytest
pytest.importorskip("pydantic")
pytestmark = pytest.mark.heavy
from typer.testing import CliRunner

import sys
import types

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
sys.modules.setdefault(
    "core.crypto",
    types.SimpleNamespace(
        generate_keypair=lambda *a, **k: None,
        verify_signature=lambda *a, **k: True,
    ),
)
sys.modules.setdefault(
    "jsonschema", types.SimpleNamespace(validate=lambda *a, **k: None)
)
import sdk.nn_models as _nn  # noqa: E402

sys.modules.setdefault("sdk.cli.nn_models", _nn)
sys.modules.setdefault(
    "core.config",
    types.SimpleNamespace(settings=types.SimpleNamespace(model_dump=lambda: {})),
)

from sdk.cli.main import app  # noqa: E402


def test_tools_list(monkeypatch):
    from sdk.cli.commands import tools as tools_cmd

    class DummyMgr:
        def list_plugins(self):
            return ["fs"]

        def get(self, name):
            return object()

    monkeypatch.setattr(tools_cmd, "PluginManager", lambda: DummyMgr())
    runner = CliRunner()
    result = runner.invoke(app, ["tools", "list"])
    assert result.exit_code == 0
    assert "fs" in result.stdout


def test_tools_inspect(monkeypatch):
    from sdk.cli.commands import tools as tools_cmd

    class DummyMgr:
        def list_plugins(self):
            return []

        def get(self, name):
            if name == "fs":
                return DummyMgr()
            return None

    monkeypatch.setattr(tools_cmd, "PluginManager", lambda: DummyMgr())
    runner = CliRunner()
    result = runner.invoke(app, ["tools", "inspect", "fs"])
    assert result.exit_code == 0
    assert "DummyMgr" in result.stdout
