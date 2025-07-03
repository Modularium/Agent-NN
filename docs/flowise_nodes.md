# Flowise Nodes for Agent-NN

Dieses Dokument listet Flowise-Komponenten, die Agent-NN bereitstellt. Jede Node kann in Flowise importiert werden und kommuniziert über HTTP mit dem jeweiligen Service.

## ListAgents

* **Typ:** Utility
* **Datei:** `integrations/flowise-nodes/ListAgents.ts`
* **Beschreibung:** Gibt alle registrierten Agenten des Registry-Dienstes zurück.
* **Parameter:**
  - `endpoint` (string): Basis-URL des Agent Registry Service.
  - `path` (string): API-Pfad, standardmäßig `/agents`.
  - `headers` (object): optionale HTTP-Header.
  - `timeout` (number): Timeout in Millisekunden.
* **Rückgabe:** JSON-Objekt mit Agenteninformationen oder Fehlermeldung.

Beispielaufruf:

```ts
const node = new ListAgents('http://localhost:8000');
const agents = await node.run();
```
