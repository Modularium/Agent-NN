# Modell-Tracking mit MLflow

Dieses Dokument fasst die bestehende Nutzung von MLflow im Projekt zusammen und beschreibt die neuen Schnittstellen der SDK und CLI.

## Aktueller Stand

Mehrere Trainingsmodule in `nn_models/` initialisieren ein `mlflow`-Experiment und loggen Parameter sowie Metriken während des Trainings:

- `advanced_training.py` setzt ein Experiment "advanced_training" und loggt Parameter und Metriken pro Epoch.
- `dynamic_architecture.py` nutzt "architecture_optimization" und protokolliert Validierungsmetriken.
- `multi_task_learning.py` verwendet "multi_task_learning".
- `training_infrastructure.py` startet Runs im Experiment "distributed_training".
- `online_learning.py` schreibt Updates unter "online_learning".
- Auch `managers/model_manager.py` beginnt MLflow-Runs beim Laden von Modellen.

Beispiele für den Einsatz sind in `advanced_training.py` zu sehen:
```python
self.experiment = mlflow.set_experiment("advanced_training")
with mlflow.start_run(
    experiment_id=self.experiment.experiment_id
) as run:
    mlflow.log_params({"num_epochs": num_epochs})
    mlflow.log_metrics({...}, step=epoch)
```
【F:nn_models/advanced_training.py†L187-L367】

Die Hilfsfunktionen im Verzeichnis `mlflow_integration/` bieten eine optionale Kapselung des Loggings. `experiment_tracking.py` stellt eine Klasse `ExperimentTracker` bereit, die Run-Namen erzeugt, Parameter und Metriken loggt und nach dem besten Run suchen kann.

Tracking-URI und Experimente werden über `config.MLFLOW_TRACKING_URI` gesteuert. Fehlt eine Verbindung, fällt MLflow auf ein lokales Verzeichnis `mlruns/` zurück.

## Erweiterungen

Für Phase 1.9.1 wurde die SDK um eine `ModelManager`-Klasse in `sdk.nn_models` ergänzt. Sie ermöglicht das Auflisten von Experimenten, das Anzeigen von Run-Informationen und das Laden von Modellen aus der Registry. Die CLI erhält neue Unterbefehle unter `agentnn model`, um diese Funktionen bequem aufzurufen.

Die Trainingsfunktionen rufen künftig `mlflow.start_run`, `mlflow.log_params` und `mlflow.log_metrics` automatisch auf. Das Tracking-URI kann über die Umgebungsvariable `MLFLOW_TRACKING_URI` oder die Datei `.env` angepasst werden.
