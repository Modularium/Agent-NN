import mlflow
from config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def log_experiment(metrics: dict, params: dict):
    with mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)
        for mk, mv in metrics.items():
            mlflow.log_metric(mk, mv)
