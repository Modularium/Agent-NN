# TODO-Liste für Agent-NN Entwicklungsplan

## Phase 1: Codebase-Konsolidierung & Architektur-Refaktor

### 1.1 Architektur-Grundlagen
- [x] Legacy-Code identifizieren und Struktur vorbereiten: Alte Module analysieren und in `archive/` verschieben.
- [x] ModelContext-Datentyp definieren: zentralen Typ für Modell- und Kontextinformationen erstellen.
- [x] Grundservices als Microservices anlegen: Dispatcher-, Registry-, Session-Manager-, Vector-Store- und LLM-Gateway-Service.
- [x] Docker-Umgebung einrichten: `docker-compose.yml` für alle Services.
- [x] Ersten End-to-End-Test durchführen: Dummy-Request über alle Services schleusen.

### 1.2 Kernservices & Kontext-Integration
- [ ] MCP-SDK integrieren & Basis-Routing implementieren.
- [ ] LLM-Gateway-Anbindung umsetzen.
- [ ] VectorStore-Service integrieren.
- [ ] Session-Manager mit persistentem Kontext.

### 1.3 Infrastruktur & Sicherheit
- [ ] Monitoring- und Logging-Infrastruktur einführen.
- [ ] Sicherheits-Layer implementieren.
- [ ] Persistente Speicherpfade & Konfiguration festlegen.

### 1.4 Entwickler-Tools & Release-Vorbereitung
- [ ] Developer-SDK und CLI bereitstellen.
- [ ] Erste Beta-Release vorbereiten.

## Phase 2: Feature-Integration & Harmonisierung

### 2.1 Meta-Learning-Integration
- [x] MetaLearner aktivieren.
- [ ] AutoTrainer und Feedback-Schleife implementieren.
- [x] Capability-basiertes Routing prototypisieren.
- [ ] Evaluationsmetriken sammeln.

### 2.2 LLM-Provider-Integration & konfigurierbare KI
- [ ] Provider-System erweitern.
- [ ] Dynamische Konfiguration über `llm_config.yaml`.
- [ ] SDK-Beispiele und Tests hinzufügen.
- [x] OpenHands API Anbindung produktiv aktivieren.

### 2.3 System-Harmonisierung & Legacy-Migration
- [ ] Bestehende Funktionen angleichen.
- [ ] Konsistenz und Abwärtskompatibilität prüfen.
- [ ] Legacy-Code schrittweise stilllegen.

## Phase 3: Konsolidiertes Frontend & User Experience

### 3.1 Vollständige Frontend-Implementierung
- [ ] Alle UI-Seiten/Panels fertigstellen.
- [ ] Frontend mit Backend-APIs verbinden.
- [ ] Einheitliche Benutzeroberfläche konsolidieren.

### 3.2 UX-Optimierungen
- [ ] Authentifizierungs-Flow verbessern.
- [ ] Fehlerbehandlung und Ladeindikatoren.
- [ ] Responsive Design & Dark-Mode sicherstellen.
- [ ] Zugänglichkeit (Accessibility) erhöhen.
- [ ] Formularvalidierung und Usability verbessern.
- [ ] (Optional) Erweiterte UI-Funktionen integrieren.

## Phase 4: Testabdeckung, CI/CD und Dokumentation

### 4.1 Testabdeckung & CI-Pipeline
- [ ] Umfassende Test-Suite entwickeln.
- [ ] Regressionsprüfung & Bugfixing.
- [ ] Continuous Integration einrichten.
- [ ] Qualitätsmetriken beobachten.

### 4.2 Deployment & Dokumentation
- [ ] Deployment-Skripte finalisieren.
- [ ] Dokumentation aktualisieren.
- [ ] SDK- und API-Dokumentation ergänzen.
- [ ] Abschließender Usability-Test.
- [ ] Release-Vorbereitung.
- [ ] Legacy-Code endgültig archivieren.
