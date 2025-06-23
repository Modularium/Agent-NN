# Changelog

All notable changes to this project are documented in this file.

## Unreleased
- Roadmap und Codex-Konfiguration an Entwicklungsplan angepasst
- Neue Phasenbeschreibungen in `.codex.json`
- Aufgabenliste in `codex.tasks.json` aktualisiert
- Basisdatenstrukturen `ModelContext` und `TaskContext` hinzugefügt
- Microservice-Gerüste für Dispatcher, Registry, LLM-Gateway, Vector-Store,
  Session-Manager und Example-Agent erstellt
- MCP Python SDK eingebunden und Dispatcher-Routing implementiert
- Sicherheitslayer mit Token-Auth, Rate-Limiting und Payload-Checks umgesetzt
- Developer SDK und CLI eingeführt (#phase-1-sdk)
- VectorStore-Service integriert, LLM-Gateway um `/embed` erweitert
- Sample-Agent nutzt jetzt semantische Suche
- SessionManager speichert ModelContext-Historien und Dispatcher/Worker
  unterstützen optionale `session_id`
- Einheitliches JSON-Logging und Prometheus-Metriken für alle Services
- Persistente Speicherpfade über `.env` konfigurierbar
- VectorStore und SessionManager unterstützen Dateispeicherung

## v1.0.0-beta
- Deployment-Skripte und Dokumentation fertiggestellt
- Erste Beta-Version mit stabilem SDK und CLI
- HTTP-Schnittstellen eingefroren

## v1.0.0
- Deployment-Skripte und Dokumentation fertiggestellt
- Erste stabile Version
- SDKs und Release-Dokumente
- Production Docker Compose

## v0.9.0-mcp
- Abschluss der MCP-Migration
- Dokumentation überarbeitet
- Docker-Compose für Kernservices
- Tests und Linting eingerichtet

## Frühere Phasen
- Phase 1: Architektur-Blueprint
- Phase 2: Kernservices
- Phase 3: Wissens- und LLM-Services
- Phase 4: Qualitätssicherung
