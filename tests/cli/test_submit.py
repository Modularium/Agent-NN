import pytest
pytest.importorskip("pydantic")
from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
from typer.testing import CliRunner
import sys
import types
from contextlib import contextmanager


@contextmanager
def _dummy_run(*args, **kwargs):
    yield types.SimpleNamespace(
        info=types.SimpleNamespace(run_id="dummy", status="FINISHED")
    )


sys.modules.setdefault(
    "mlflow",
    types.SimpleNamespace(
        start_run=_dummy_run,
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


class DummyResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def post(self, url, json=None, headers=None):
        return DummyResp({"ok": True})

    def get(self, url, headers=None):
        return DummyResp({"items": []})


def test_cli_submit(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda base_url: DummyClient())
    runner = CliRunner()
    result = runner.invoke(app, ["submit", "test"])
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_cli_ask(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda base_url: DummyClient())
    runner = CliRunner()
    result = runner.invoke(app, ["ask", "hi", "--task-type=dev"])
    assert result.exit_code == 0
    assert "ok" in result.stdout
