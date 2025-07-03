
# Flowise-Flow für ein 16‑Agenten-Team mit OpenHands

Wir erstellen einen Flowise-Workflow, der 16 lokal laufende OpenHands-Agenten (Ports 3001–3016) als Spezialisten ansteuert. Neben den bisherigen Rollen wie „Frontend“, „Backend“ oder „Dokumentation“ sind zusätzliche Experten vertreten. Dazu zählen Android‑ und iOS‑Entwicklung, plattformübergreifende App-DevOps, Spezialisten für neuronale Netze, allgemeines ML/DL sowie LLM‑DevOps.

**Agentenübersicht:**

- Frontend Engineer
- Backend Engineer
- DevOps Specialist
- Security Analyst
- QA Tester
- Technical Writer
- Code Reviewer
- Release Manager
- Performance Optimizer
- UX Designer
- Android App DevOps
- iOS App DevOps
- Cross‑Platform App DevOps
- Neural Network Expert
- ML/DL Expert
- LLM DevOps

Flowise ist ein Open-Source Low‑Code-Werkzeug, mit dem man KI‑Workflows visuell zusammenstellen kann. OpenHands ist ein Open-Source-Agentensystem, das Entwickleraufgaben per natürlicher Sprache ausführt. Im Flowise‑Flow richten wir folgende Komponenten ein:

## Setup

1. Stelle sicher, dass Python 3.10, Docker und Redis installiert sind.
2. Installiere die Basis-Abhängigkeiten mit

 ```bash
  pip install -r requirements.txt
  ./install_openhands_deps.sh  # optional
  ```


   Alternativ kann `pip install -r requirements-openhands.txt` ausgeführt werden.

   Die Ports der Agenten lassen sich über die Umgebungsvariable `OPENHANDS_AGENT_PORTS` steuern (Standard `3001-3016`).

3. Starte alle OpenHands-Agenten (Standardports 3001‑3016) und Flowise.

In Test- oder CI-Umgebungen ohne Docker kann der OpenHands-API-Server durch
`tests/mocks/fake_openhands.py` simuliert werden. Die zugehörigen Tests
überspringen fehlende Abhängigkeiten automatisch.


* **User-Input-Node:** Ein Texteingabe-Node (“User Prompt”), in dem der Nutzer die Aufgabenbeschreibung eingibt (z.B. „Implementiere ein Login-Formular in React“). Dieser Text wird als Variable im Flow gespeichert (z.B. in `state.user_task`).

* **Auswahl-Node (Dropdown/Checkbox):** Ein Auswahl- bzw. Switch-Node, mit dem der Nutzer eine oder mehrere Ziel-Instanzen auswählt (z.B. „Frontend“, „Backend“, „Dokumentation“). Dies kann über ein Dropdown oder Kontrollkästchen realisiert werden. Die Auswahl wird in einer State-Variable (z.B. `state.selected_agents`) festgehalten.

* **Entscheidungs- / Bedingungs-Nodes:** Für jede Agenten-Option (Frontend, Backend etc.) fügen wir einen „Condition“-Node ein. Jeder prüft etwa, ob der jeweilige Agent in der Auswahl enthalten ist (z.B. Bedingung: `state.selected_agents` enthält „Frontend“). Auf „True“ verzweigt der Flow zu den Tool-Nodes dieses Agents. Flowise kann dabei parallele Pfade ausführen, sodass mehrere Agenten gleichzeitig abgefragt werden können. In jedem Pfad leiten wir also die Eingabe zur entsprechenden Instanz weiter.

* **Tool-Nodes (HTTP POST):** In jedem aktivierten Zweig verwenden wir einen „Request“-Tool-Node, der eine POST-Anfrage an die OpenHands-API des jeweiligen Agents sendet. Beispiel: Für den Frontend-Agenten konfiguriert man einen Request-Node mit URL

  ```
  http://localhost:3001/api/conversations
  ```

  Methode `POST`. Im HTTP-Header trägt man falls nötig einen Autorisierungs-Token ein (z.B. `Authorization: Bearer <token>`) sowie `Content-Type: application/json`. Der JSON-Body enthält mindestens das Feld `initial_user_msg` mit der Nutzeraufgabe. Beispiel-Body:

  ```json
  {
    "initial_user_msg": "{{state.user_task}}"
  }
  ```

  (`{{state.user_task}}` steht hier für die vom Nutzer eingegebene Aufgabenbeschreibung.) Dies entspricht dem OpenHands-API-Standard zum Starten einer neuen Konversation. Die Antwort enthält u.a. eine `conversation_id`, die wir per **Set-Variable-Node** speichern (z.B. in `state.conversation_id`). (Beispiel-Antwort: `{"status":"ok","conversation_id":"abc1234"}`.)

* **Tool-Nodes (HTTP GET Trajectory):** Nachdem die Konversation gestartet ist, holen wir mit einem weiteren Request-Node die Ergebnisse ab. Dazu senden wir eine GET-Anfrage an

  ```
  http://localhost:3001/api/conversations/{conversation_id}/trajectory
  ```

  (Port passend zur Instanz). Wir fügen denselben Autorisierungs-Header ein. Die Antwort liefert die Aktions-Trajektorie des Agenten in JSON-Form (Feld `trajectory`). Beispiel (vereinfacht):

  ```json
  {
    "trajectory": [
      { /* Aktion 1 */ },
      { /* Aktion 2 */ }
    ]
  }
  ```

  Dies entspricht dem Dokumentationsbeispiel für den Trajectory-Endpoint.

* **Tool-Nodes (GET List-Files und Select-File):** Optional kann man nach Abschluss der Aufgaben auch die produzierten Dateien abrufen. Mit `GET /api/conversations/{id}/list-files` bekommen wir ein JSON-Array mit Dateipfaden. Beispiel:

  ```
  curl -X GET "http://localhost:3001/api/conversations/abc1234/list-files"
  ```

  liefert etwa `["/workspace/log.txt","/workspace/output/result.json"]`. Anschließend kann man mit `GET /api/conversations/{id}/select-file?file=<pfad>` den Inhalt einer bestimmten Datei auslesen. Beispiel:

  ```
  curl -X GET "http://localhost:3001/api/conversations/abc1234/select-file?file=/workspace/log.txt"
  ```

  Dies liefert etwa `{"code":"<Inhalt der Datei>"}`. Die Fließtext-Antworten (Ergebnis-Log, Code-Log) lassen sich dann im Flowise-Output-Node anzeigen.

* **Ausgabe-Node:** Schließlich bündeln wir die Ergebnisse für den Nutzer. Ein “Text Output”-Node (bzw. der Assistant-Output) zeigt die Agentenantworten, Logs oder Datei-Inhalte an. Man könnte z.B. die JSON-Antworten formatieren oder einfach den Textinhalt der Logdatei direkt ausgeben.

Die gesamte Logik sieht so aus: Nutzer gibt Aufgabe und Agenten-Auswahl ein. Anhand der Auswahl verzweigt Flowise in parallele Teil-Flows: Jeder aktivierte Teil ruft das jeweilige OpenHands-Agenten-API auf (mittels Flowise-Tools) und holt Trajektorie bzw. Ergebnisse ab. In Flowise geschieht die HTTP-Anfrage über die **Request-Tool-Nodes** (GET/POST). Zum Beispiel wählt der Flowise-Agent bei einer GET-Anfrage den „GET-Tool“ aus und sendet sie an den konfigurierten Endpunkt. Parallel dazu können mehrere solcher Tools ausgeführt werden, um z.B. Frontend- und Backend-Agenten gleichzeitig anzusprechen.

**Beispielkonfiguration (Tool-Nodes):**

* *POST-Request an OpenHands:*

  ```json
  {
    "tool": "Request Post",
    "method": "POST",
    "url": "http://localhost:3001/api/conversations",
    "headers": {
      "Authorization": "Bearer <API-Token>",
      "Content-Type": "application/json"
    },
    "body": {
      "initial_user_msg": "{{state.user_task}}"
    }
  }
  ```

  (Analog für die Ports 3002–3016 der anderen Instanzen.) Dieses JSON entspricht dem cURL-Beispiel aus der OpenHands-Doku.

* *GET-Request Trajectory:*

  ```json
  {
    "tool": "Request Get",
    "method": "GET",
    "url": "http://localhost:3001/api/conversations/{{state.conversation_id}}/trajectory",
    "headers": {
      "Authorization": "Bearer <API-Token>"
    }
  }
  ```

  und ähnlich für `/list-files` oder `/select-file`. Die Pfade und IDs setzt Flowise durch Platzhalter (`{{state.conversation_id}}`) aus der gespeicherten Variable ein.

* *Set-Variable-Node:* Direkt nach jedem POST-Tool-Node nutzen wir einen **Set-Variable-Node**, um den Wert `conversation_id` aus der Antwort in der State zu sichern. Danach kann jeder nachfolgende GET-Tool-Node diesen State-Wert in der URL verwenden.

* *Output-Node:* Am Ende jedes Pfades hängt ein Ausgabe-Node, der die abgerufenen Texte oder JSON-Felder an den User ausgibt (z.B. Log-Inhalte aus `select-file`).

**Beispiel für einen Custom Tool-Node (skalierbar):** Man kann auch eine benutzerdefinierte Komponente schreiben, die dynamisch Port/Instanz anspricht. Z.B. ein Custom-Node mit Parameter `agent_port` und `apiKey`, der dann die obige POST-Anfrage an `http://localhost:${agent_port}/api/conversations` führt. Pseudocode (Node.js) könnte so aussehen:

```js
const fetch = require('node-fetch');
async function openHandsRequest(agent_port, apiKey, userTask) {
  const url = `http://localhost:${agent_port}/api/conversations`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ initial_user_msg: userTask })
  });
  const data = await res.json();
  // conversation_id in State speichern, etc.
}
```

Ein solcher Custom Tool Node würde dann flexibel jeden Agenten-Port ansteuern.

**Quellen:** Die Konfiguration orientiert sich an der offiziellen OpenHands-API-Dokumentation (z.B. POST `/api/conversations` mit `"initial_user_msg"`, GET `/trajectory`, GET `/list-files`, GET `/select-file`). Flowise-Tool-Nodes für HTTP-Anfragen sind dokumentiert – der Flowise-Agent wählt z.B. ein GET-Tool aus und sendet einen GET-Request an den konfigurierten Endpunkt. Zudem unterstützt Flowise parallele Branches, sodass mehrere Agenten-Aufrufe simultan ablaufen können. Alle Konfigurationseinstellungen (URL, Methode, Header, JSON-Body) lassen sich in den Request-Tool-Nodes angeben und verwenden Platzhalter für dynamische Werte (z.B. Nutzer-Input und gespeicherte `conversation_id`).
