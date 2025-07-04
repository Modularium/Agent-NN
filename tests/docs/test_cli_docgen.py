from pathlib import Path
import sys
import types

from typer.testing import CliRunner

import sdk.nn_models as _nn

# stub heavy dependencies
sys.modules.setdefault(
    "mcp",
    types.SimpleNamespace(
        types=types.SimpleNamespace(BaseModel=object, Field=lambda *a, **k: None)
    ),
)
sys.modules.setdefault(
    "agentnn.session.session_manager", types.SimpleNamespace(SessionManager=object)
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

from tools.cli_docgen import collect_cli_info  # noqa: E402
from sdk.cli.main import app  # noqa: E402


def test_all_commands_documented() -> None:
    groups, root_cmds = collect_cli_info()
    documented = {
        line.split("|")[1].strip().strip("`")
        for line in Path("docs/cli.md").read_text().splitlines()
        if line.startswith("| `")
    }
    expected = {name for name, _, _ in groups} | {name for name, _, _ in root_cmds}
    assert expected <= documented


def test_cli_help_contains_commands() -> None:
    groups, _ = collect_cli_info()
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name, _, _ in groups:
        assert name in result.stdout
