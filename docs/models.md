# Modellübersicht

Die neuronalen Modelle befinden sich im Ordner `nn_models/`.

| Modelldatei | Beschreibung |
|-------------|-------------|
| `agent_nn.py` | Enthält das Kernnetz zur Bewertung von Aufgaben und zur Generierung von Feature-Vektoren. |
| `model_init.py` | Stellt Hilfsfunktionen bereit, um Modelle zu initialisieren oder zu laden. |
| `meta_learner.py` | Implementiert ein einfaches Meta-Lernverfahren zur Agentenbewertung. |
| `dynamic_architecture.py` | ⚠️ Migrated Legacy-Modell für anpassbare Netzarchitekturen. |
| `multi_task_learning.py` | ⚠️ Migrated Legacy-Modell für Mehrfachaufgaben-Lernen. |
| `online_learning.py` | ⚠️ Migrated Legacy-Modell für fortlaufendes Lernen. |
| `advanced_training.py` | ⚠️ Migrated Legacy-Modell mit Hierarchien und Attention. |
| `distributed_training.py` | ⚠️ Migrated Legacy-Modell für verteiltes Training. |
| `parallel_models.py` | ⚠️ Migrated Legacy-Modell zur Modellparallelisierung. |
| `training_infrastructure.py` | ⚠️ Werkzeuge für Legacy-Trainingsinfrastruktur. |
| `agent_nn_v2.py` | ⚠️ Frühere Variante des AgentNN-Netzes. |
| `model_security.py` | ⚠️ Hilfsmodule für Sicherheitsprüfungen beim Laden von Modellen. |

Modelle können über den `ModelManager` geladen und anschließend als Tools registriert werden. Eigene Modelle lassen sich durch Ableiten der Basisklassen einbinden und über die Konfigurationsdateien aktivieren.
