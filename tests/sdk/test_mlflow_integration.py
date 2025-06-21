from sdk.nn_models import ModelManager


def test_list_experiments(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path))
    mgr = ModelManager()
    exps = mgr.list_experiments()
    assert isinstance(exps, list)
