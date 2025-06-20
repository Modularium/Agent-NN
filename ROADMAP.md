# MVP Roadmap – Agent-NN

Diese Roadmap zeigt den Weg zum **Minimal Viable Product** für Agent-NN. Die Entwicklung ist in Phasen unterteilt; innerhalb jeder Phase gibt es konkrete Aufgaben. Abgehakte (✅) Punkte sind bereits erledigt, offene (⬜) werden noch bearbeitet.

## Phase 1: Analyse  
- ⬜ **Codebase analysieren:** Architektur und bestehende Komponenten durchleuchten, fehlende Features identifizieren.  
- ⬜ **Verbesserungspotential dokumentieren:** Schwachstellen in Design/Code notieren (für Planung relevant).  
- ⬜ **Architektur-Bericht erstellen:** Kurzbeschreibung der aktuellen Struktur und offenen Punkte (Output: `docs/architecture/analysis.md`).  

## Phase 2: Planung  
- ⬜ **Roadmap erstellen/aktualisieren:** (dieses Dokument) Phasen und Tasks definieren, basierend auf Analyseergebnissen.  
- ⬜ **Tasks spezifizieren:** Klare Beschreibung der umzusetzenden Features (LOH-Agent, Setup-Agent, Framework) und Verbesserungen (Supervisor vervollständigen, Logging etc.).  
- ⬜ **Akzeptanzkriterien festlegen:** Wann gilt jeder Task als abgeschlossen? (z.B. bestimmter Test grün).  
- ⬜ **AGENTS.md anlegen:** Rollen und Entwicklungsrichtlinien definieren, um gleichbleibende Qualität sicherzustellen.  

## Phase 3: Umsetzung (Development)  
- ⬜ **SupervisorAgent abschließen:** Implementierung fertigstellen, inkl. Task-Delegation (Kriterium: Test für SupervisorAgent besteht).  
- ⬜ **LOH-Agent implementieren:** Neuen Worker-Agent (LOH) gemäß Spezifikation implementieren.  
- ⬜ **Agent-Setup implementieren:** neuen Agenten für Initialisierungs-/Setup-Aufgaben einfügen (falls vorgesehen).  
- ⬜ **Agent-Framework implementieren:** Agent zur Verwaltung des Frameworks oder Meta-Agent (genaue Anforderungen klären und umsetzen).  
- ⬜ **Neural Network Routing aktivieren:** `NNManager` und zugehörige Modelle funktionsfähig machen (inkl. Einbindung in Supervisor).  
- ⬜ **VectorStore & Knowledge-Base integrieren:** Dokumenten-Embedding und -Retrieval vollständig nutzbar machen (Test: WorkerAgent kann Wissensdaten abfragen).  
- ⬜ **Logging & Error-Handling einführen:** zentrales Logging (`logging_util.py`) konfigurieren; Fehlerfälle werden geloggt und ggf. von Supervisor gehandhabt.  
- ⬜ **Smoke-Tests nach Features:** Nach jeder größeren Implementierung einmal das System end-to-end testen (manuell/automatisiert), um grobe Fehler sofort zu fixen.  

## Phase 4: Qualitätssicherung (Testing)  
- ⬜ **Unit-Tests schreiben:** Tests für alle neuen oder geänderten Module (SupervisorAgent, neue Agents, Manager etc.).  
- ⬜ **Testlauf erfolgreich:** Alle Tests (inkl. bestehender `test_agent_manager.py`, `test_agent_nn.py` etc.) grün.  
- ⬜ **Linting/Typing clean:** Keine Linter-Warnungen oder Typfehler mehr (Black, Flake8, isort, mypy ausgeführt und zufriedenstellend).  
- ⬜ **Integrationstest:** Kompletten Ablauf mit Beispiel-Task testen: Eingabe → Chatbot/Supervisor → korrekter Worker → Ausgabe. Ergebnis ist sinnvoll und korrekt.  
- ⬜ *(optional)* **Performance ok:** Grundlegende Performance-Metriken gemessen (z.B. Antwortzeit, Ressourcenauslastung) und im Auge behalten.  

## Phase 5: Dokumentation & Abschluss  
- ⬜ **Benutzer-Doku fertig:** Anleitungen für Installation, Konfiguration und Nutzung in `docs/BenutzerHandbuch` sind komplett.  
- ⬜ **Entwickler-Doku fertig:** Technische Doku (Architektur, API, CLI, Dev Guide) in `docs/` vervollständigt. Architekturdiagramm erstellt.  
- ⬜ **CONTRIBUTING Guide fertig:** Richtlinien für Beiträge (Code Style, Testing, Workflow) erstellt/aktualisiert.  
- ⬜ **README/Projektinfos aktualisiert:** README.md reflektiert finalen MVP-Status; Roadmap (dieses Dokument) zeigt alle erledigten Punkte.  
- ⬜ **Release vorbereitet:** (Falls nötig) Docker-Container gebaut und lauffähig, Versionsnummer erhöht, Tag gesetzt.

---

**Legende:** ✅ = erledigt, ⬜ = offen/zu tun.  
Sobald alle Punkte einer Phase erledigt sind, wechselt der Codex-Agent automatisch in die nächste Phase. Dieses iterative Vorgehen gewährleistet eine systematische Fertigstellung des MVP mit minimalen Rückschlägen.
## Konkrete MVP-Aufgaben

Die folgenden Punkte fassen alle noch fehlenden Arbeiten zusammen. Zu jedem Task sind Ziel und Akzeptanzkriterien aufgeführt.

1. ⬜ **LOH-Agent implementieren**
   - *Ziel:* Spezialisierten LOH-Agent anlegen, der LOH-relevante Aufgaben ausführt.
   - *Abhängigkeiten:* Funktionierender SupervisorAgent.
   - *Akzeptanzkriterien:* Modul `agents/loh_agent.py` existiert und der Test `tests/test_loh_agent.py` besteht.
2. ⬜ **Setup-Agent & Agent-Framework entwickeln**
   - *Ziel:* Einen Setup-Agenten und ein Basis-Framework erstellen, um neue Worker automatisiert einzurichten.
   - *Akzeptanzkriterien:* `tests/test_setup_agent.py` deckt die Initialisierung ab; Ausführung erstellt einsatzbereite Worker ohne Fehler.
3. ⬜ **SupervisorAgent testen**
   - *Ziel:* Kernlogik des Supervisors per Unit-Test absichern.
   - *Akzeptanzkriterien:* Datei `tests/test_supervisor_agent.py` vorhanden und grün.
4. ⬜ **Zentrales Logging einrichten**
   - *Ziel:* Einheitliches Logging in allen Modulen mit `LoggerMixin`.
   - *Akzeptanzkriterien:* Logs erscheinen im definierten Format; Fehler werden in `logs/` protokolliert.
5. ⬜ **Integrationstests für Agent-Kommunikation**
   - *Ziel:* Sicherstellen, dass Chatbot → Supervisor → Worker fehlerfrei funktioniert.
   - *Akzeptanzkriterien:* Ein End-to-End-Test in `tests/test_integration.py` läuft erfolgreich durch.
6. ⬜ **CLI und API vervollständigen**
   - *Ziel:* Fehlende Befehle und Endpunkte implementieren, sodass typische Workflows per Kommandozeile oder HTTP möglich sind.
   - *Akzeptanzkriterien:* Befehle starten Agents ohne Exceptions; API-Test `tests/test_api.py` besteht.

### Offene Abhängigkeiten
- `transformers` und `langchain_openai` sind aktuell nicht installiert. Vor den Tests müssen diese Pakete verfügbar sein.
