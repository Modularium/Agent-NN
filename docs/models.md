# Modellübersicht

Die neuronalen Modelle befinden sich im Ordner `nn_models/`.

| Modelldatei | Beschreibung |
|-------------|-------------|
| `agent_nn.py` | Enthält das Kernnetz zur Bewertung von Aufgaben und zur Generierung von Feature-Vektoren. |
| `model_init.py` | Stellt Hilfsfunktionen bereit, um Modelle zu initialisieren oder zu laden. |
| `meta_learner.py` | Implementiert ein einfaches Meta-Lernverfahren zur Agentenbewertung. |

Modelle können über den `ModelManager` geladen und anschließend als Tools registriert werden. Eigene Modelle lassen sich durch Ableiten der Basisklassen einbinden und über die Konfigurationsdateien aktivieren.
