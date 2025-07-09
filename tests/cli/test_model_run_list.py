import pytest
pytest.importorskip("pydantic")
from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
from typer.testing import CliRunner
from sdk.cli.main import app


def test_model_runs_list(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["model", "runs-list"])
    assert result.exit_code == 0
