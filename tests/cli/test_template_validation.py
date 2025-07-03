from pathlib import Path
import sys
import types

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
sys.modules.setdefault(
    "core.config",
    types.SimpleNamespace(settings=types.SimpleNamespace(model_dump=lambda: {})),
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

from typer.testing import CliRunner
import pytest

from sdk.cli.main import app
from sdk.cli import config as cli_config


def _patch_config(monkeypatch, tmp_path: Path) -> Path:
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    default = tpl_dir / "session_template.yaml"
    default.write_text("agents: []\ntasks: []")

    def load():
        return cli_config.CLIConfig(
            default_session_template=str(default),
            output_format="table",
            log_level="INFO",
            templates_dir=str(tpl_dir),
        )

    monkeypatch.setattr(cli_config.CLIConfig, "load", classmethod(lambda cls: load()))
    return tpl_dir


@pytest.mark.unit
def test_template_validate(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    _patch_config(monkeypatch, tmp_path)
    cfg = tmp_path / "agent.yaml"
    cfg.write_text("id: demo\nrole: assistant")
    result = runner.invoke(app, ["template", "validate", str(cfg)])
    assert result.exit_code == 0
    assert "G\u00fcltig" in result.stdout


@pytest.mark.unit
def test_template_validate_fail(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    _patch_config(monkeypatch, tmp_path)
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("role: assistant")
    result = runner.invoke(app, ["template", "validate", str(cfg)])
    assert result.exit_code != 0
    assert "Ung\u00fcltig" in result.stdout
