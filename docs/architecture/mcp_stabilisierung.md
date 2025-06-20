# MCP Stabilisierung

Diese Phase fokussiert sich auf Tests, Logging und ein lauffähiges Docker-Setup.

## Qualitätssicherung
- Neue Unit-Tests decken Worker-Services, Vector Store, LLM-Gateway und Dispatcher ab.
- `tests/ci_check.sh` führt Ruff, Mypy und Pytest mit Coverage aus.

## Resilienz
- Interne REST-Aufrufe besitzen Timeouts und Fehlerbehandlung.
- Worker-Aufrufe werden im Dispatcher protokolliert; Fehler werden sauber zurückgegeben.

## Deployment-Vorbereitung
- Alle Dienste besitzen minimale Docker-Container.
- `scripts/build_and_start.sh` baut Images und startet die Compose-Umgebung.

Weitere Schritte beinhalten zentrale Log-Aggregation und Monitoring über Prometheus.
