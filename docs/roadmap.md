# Agent-NN Roadmap

Dieses Dokument beschreibt empfohlene Weiterentwicklungen nach Version 1.0.0.

## Phase 5 – Erweiterte Lernmechanismen
- Feinjustierung vorhandener Modelle via Reinforcement Learning und Few-Shot-Training.
- Aufbau einer Föderationsschicht, um mehrere Agent-NN-Instanzen zu koordinieren.
- Unterstützung für komplexe Multi-Agent-Aufgaben mit geteiltem Kontext.

## Phase 6 – Skalierung & Federation
- Lastverteilung und horizontale Skalierung der Worker-Dienste.
- Federation mit externen Agent-Plattformen über standardisierte APIs.
- Ausbau der Überwachungs- und Governance-Werkzeuge.

## Teamweitergabe & Onboarding
- Neue Entwickler starten am besten mit `docs/deployment.md` und `docs/maintenance.md`.
- Kernkomponenten befinden sich in `services/` und `mcp/`; die React-Oberfläche unter `frontend/agent-ui`.
- Für Tests dient `tests/test_all.py` als Einstiegspunkt.

## Strategische Ziele
- Minimierung technischer Schulden durch klare Modultrennung.
- Weitere Automatisierung des Trainingsprozesses und Self-Healing der Dienste.
- Dokumentierte Entscheidungsprozesse in `docs/architecture/` fortführen.

