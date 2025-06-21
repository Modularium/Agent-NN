# Roadmap – MCP Refactoring

Diese Datei orientiert sich am *Entwicklungsplan für das Agent-NN Framework*. Die Migration wird schrittweise in vier Phasen umgesetzt. Jedes Element ist mit Tags versehen, damit Fortschritte leicht nachvollziehbar sind.

## Phase 1 – Architektur-Migration (#phase-1 #mcp)
- [ ] **P1T1** ModelContext-Datentyp definieren
- [ ] **P1T2** Grundservices (Dispatcher, Registry, Session Manager, Vector Store, LLM Gateway) als Microservices anlegen
- [ ] **P1T3** Docker Compose für alle Services erstellen
- [ ] **P1T4** End-to-End-Test über die neuen Services
- [ ] **P1T5** MCP-SDK integrieren und Basis-Routing implementieren

## Phase 2 – Agent-NN & Meta-Learning (#phase-2)
- [ ] **P2T1** MetaLearner im NNManager aktivieren
- [ ] **P2T2** AutoTrainer und Feedbackschleife implementieren
- [ ] **P2T3** Capability-basiertes Routing in einem WorkerAgent prototypisieren
- [ ] **P2T4** Evaluationsmetriken sammeln

## Phase 3 – SDK & Provider-System (#phase-3 #sdk)
- [ ] **P3T1** Provider-Klassen und Factory im SDK entwickeln
- [ ] **P3T2** Konfigurierbare `llm_config.yaml` nutzen
- [ ] **P3T3** Beispiele und Unit-Tests für das SDK
- [ ] **P3T4** Dokumentation unter `docs/sdk/`

## Phase 4 – Testing, CI/CD und Doku (#phase-4)
- [ ] **P4T1** Vollständige pytest-Suite mit Mocks
- [ ] **P4T2** GitHub-Actions-Workflow mit ruff, black, pytest und coverage
- [ ] **P4T3** Docker-Compose und Deployment-Skripte finalisieren
- [ ] **P4T4** Dokumentation und Architekturbild aktualisieren

---

Legende: `[ ]` offen / `[x]` erledigt. Jede Phase baut auf der vorherigen auf.
