import mlflow
from sdk.nn_models import ModelManager


def test_track_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path))
    mgr = ModelManager()
    run_id = mgr.track_run("demo", {"p": 1}, {"m": 2.0}, {"t": "x"})
    info = mgr.get_run_summary(run_id)
    assert info["metrics"]["m"] == 2.0
