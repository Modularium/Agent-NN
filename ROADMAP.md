# MCP Roadmap – Agent-NN

Diese Roadmap skizziert die Migration von Agent-NN zur Modular Control Plane Architektur. Jeder Abschnitt beschreibt die wichtigsten Schritte und den Zielzustand.

## Phase 1: Architektur-Blueprint
- ✅ Schnittstellen und Verantwortlichkeiten der neuen Dienste festlegen
- ✅ Ordnerstruktur und Docker-Compose-Skelett anlegen
- ✅ Beispiel-Dispatcher, Registry und Dummy-Worker starten

## Phase 2: Kernservices
- ✅ Agent-Registry mit statischen Einträgen implementieren
- ✅ Task-Dispatcher auf Registry und Session-Manager umstellen
- ✅ Redis-basierte Session-Verwaltung bereitstellen

## Phase 3: Wissens- und LLM-Services
- ✅ Eigenständiger Vector-Store-Service mit REST-API
- ✅ LLM-Gateway-Service für OpenAI und lokale Modelle
- ✅ Worker-Services aus dem Monolithen herauslösen (Dev, OpenHands, LOH)

## Phase 4: Qualitätssicherung & Deployment-Vorbereitung
- ✅ Testsuite und Linter eingerichtet
- ✅ Logging in allen Services
- ✅ Docker-Compose Skripte erstellt

## Phase 5: Abschluss & Deployment
- ⬜ Dokumentation und README auf MCP-Architektur aktualisieren
- ⬜ Containerisierung/Docker-Compose für alle Services erstellen
- ⬜ Roadmap finalisieren und weitere Schritte planen

---

**Legende:** ⬜ offen / ✅ erledigt. Jede Phase baut auf der vorherigen auf. Nach Abschluss von Phase 5 ist Agent-NN vollständig auf die MCP-Struktur migriert.
