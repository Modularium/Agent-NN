import pytest
pytest.importorskip("pydantic")
from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
from typer.testing import CliRunner
from pathlib import Path
import sys
import types
import pytest

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
    types.SimpleNamespace(ws_server=types.SimpleNamespace(broadcast=lambda *a, **k: None)),
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
sys.modules.setdefault("sdk.cli.config", types.SimpleNamespace(SDKSettings=DummySettings))
sys.modules.setdefault(
    "core.config", types.SimpleNamespace(settings=types.SimpleNamespace(model_dump=lambda: {}))
)

from sdk.cli.main import app


@pytest.mark.unit
def test_cli_reset(monkeypatch, tmp_path):
    home = tmp_path / "home"
    data = tmp_path / "data"
    home.mkdir()
    data.mkdir()
    (home / ".agentnn").mkdir()
    db = data / "context.db"
    db.write_text("x")
    snaps = data / "snapshots"
    snaps.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("CONTEXT_DB_PATH", str(db))
    monkeypatch.setenv("SNAPSHOT_PATH", str(snaps))

    runner = CliRunner()
    result = runner.invoke(app, ["reset", "--confirm"])
    assert result.exit_code == 0
    assert "reset complete" in result.stdout
    assert not db.exists()
    assert not snaps.exists()
    assert not (home / ".agentnn").exists()
