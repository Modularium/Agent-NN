import pytest
pytest.importorskip("pydantic")
from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
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


def test_feedback_record_interactive(monkeypatch):
    from sdk.cli.commands import feedback as fb_cmd

    class DummyClient:
        def post_feedback(self, session, payload):
            return {"ok": True}

    monkeypatch.setattr(fb_cmd, "AgentClient", lambda: DummyClient())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["feedback", "record", "--interactive"],
        input="sid\n1\nGreat\n",
    )
    assert result.exit_code == 0
    assert "ok" in result.stdout
