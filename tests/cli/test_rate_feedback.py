import pytest
pytest.importorskip("pydantic")
from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
import sys
import types
import httpx
import pytest
from typer.testing import CliRunner

sys.modules.setdefault(
    "core.crypto",
    types.SimpleNamespace(
        generate_keypair=lambda *a, **k: None,
        verify_signature=lambda *a, **k: True,
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

from sdk.cli.main import app  # noqa: E402
from core.agent_profile import AgentIdentity  # noqa: E402
from core.audit_log import AuditLog  # noqa: E402


@pytest.mark.unit
def test_rate_invalid_score(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("RATING_DIR", str(tmp_path))
    # create dummy profile for to_agent so update_reputation works if called
    AgentIdentity(name="to", role="", traits={}, skills=[], memory_index=None, created_at="now").save()
    monkeypatch.setattr(AuditLog, "write", lambda self, entry: "id")
    runner = CliRunner()
    result = runner.invoke(app, ["rate", "from", "to", "--score", "1.5"])
    assert result.exit_code == 1
    assert "invalid score" in result.stdout


class DummyClient:
    def post_feedback(self, session, payload):
        request = httpx.Request("POST", "http://example")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)


@pytest.mark.unit
def test_feedback_submit_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr("sdk.cli.main.AgentClient", lambda: DummyClient())
    runner = CliRunner()
    result = runner.invoke(app, ["feedback", "submit", "sess", "--score", "1", "--comment", "hi"])
    assert result.exit_code == 1
    assert "Nicht autorisiert" in result.stdout
