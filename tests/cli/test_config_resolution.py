import pytest
pytest.importorskip("pydantic")
pytestmark = pytest.mark.heavy
from typer.testing import CliRunner
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "mcp",
    types.SimpleNamespace(
        types=types.SimpleNamespace(BaseModel=object, Field=lambda *a, **k: None)
    ),  # noqa: E501
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
                get_run=lambda run_id: types.SimpleNamespace(  # noqa: E501
                    info=types.SimpleNamespace(run_id=run_id, status="FINISHED"),
                    data=types.SimpleNamespace(metrics={}, params={}),
                ),
            )
        ),
    ),
)
sys.modules.setdefault(
    "core.config",
    types.SimpleNamespace(
        settings=types.SimpleNamespace(model_dump=lambda: {})
    ),  # noqa: E501
)
sys.modules.setdefault(
    "core.crypto",
    types.SimpleNamespace(
        verify_signature=lambda *a, **k: True,
        generate_keypair=lambda: ("pub", "priv"),
    ),
)
import sdk.nn_models as _nn  # noqa: E402

sys.modules.setdefault("sdk.cli.nn_models", _nn)

from sdk.cli.main import app  # noqa: E402
from sdk.cli import config as cli_config  # noqa: E402


def test_local_overrides_global(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    global_cfg = home / ".agentnn" / "config.toml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text("output_format = 'table'\n")

    project = tmp_path / "project"
    project.mkdir()
    local_cfg = project / "agentnn.toml"
    local_cfg.write_text("output_format = 'json'\n")

    monkeypatch.setattr(Path, "home", lambda: home)
    runner = CliRunner()

    def load():
        return cli_config.CLIConfig.load()

    monkeypatch.chdir(project)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "json" in result.stdout
