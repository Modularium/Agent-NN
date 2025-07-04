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


def test_governance_vote(monkeypatch, tmp_path):
    from sdk.cli.commands import governance as gov_cmd

    called = {}

    def fake_record(vote):
        called["vote"] = vote

    monkeypatch.setattr(gov_cmd, "record_vote", fake_record)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["governance", "vote", "p1", "--agent", "demo", "--decision", "yes"],
    )
    assert result.exit_code == 0
    assert "recorded" in result.stdout
    assert called["vote"].proposal_id == "p1"


def test_governance_log(monkeypatch):
    from sdk.cli.commands import governance as gov_cmd

    monkeypatch.setattr(gov_cmd, "load_votes", lambda p: [])
    runner = CliRunner()
    result = runner.invoke(app, ["governance", "log", "p1"])
    assert result.exit_code == 0
    assert "[]" in result.stdout
