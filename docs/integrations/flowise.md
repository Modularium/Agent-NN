# FlowiseAI Integration

[FlowiseAI](https://flowiseai.com/) bietet eine grafische Oberfläche zum Erstellen von Chatbot‑Flows. Agent‑NN lässt sich sowohl als Quelle für Antworten als auch als externer Aufrufer nutzen.

## Agent‑NN als Flowise Komponente

Unter `integrations/flowise-agentnn` liegt eine Beispielkomponente `AgentNN.ts`. Sie sendet Fragen an den Dispatcher und gibt das Ergebnis zurück. Der Kern sieht so aus:

```ts
import axios from 'axios';

export default class AgentNN {
  constructor(
    private endpoint: string,
    private taskType = 'chat',
    private headers: Record<string, string> = {},
    private timeout = 10000,
    private method: 'POST' | 'GET' | 'PUT' | 'DELETE' = 'POST',
  ) {}

  async run(payload: unknown) {
    const url = `${this.endpoint}/task`;
    const body = { task_type: this.taskType, input: payload };
    const { data } = await axios.request({
      url,
      method: this.method,
      data: body,
      headers: this.headers,
      timeout: this.timeout,
    });
    return data.result ?? data;
  }
}
```

Diese Komponente wird in Flowise eingebunden und erlaubt es, Benutzeranfragen direkt an Agent‑NN zu delegieren.

## Flowise Workflows aus Agent‑NN anstoßen

Das Plugin `flowise_workflow` ruft HTTP‑basierte Chatflows auf. Neben einem Payload können optionale Header, die HTTP-Methode und ein Timeout übergeben werden:


```python
from plugins.flowise_workflow.plugin import Plugin

plugin = Plugin()
result = plugin.execute(
    {
        "endpoint": "http://localhost:3000",
        "path": "/api/v1/predict",
        "payload": {"question": "Hi"},
        "headers": {"Authorization": "Bearer token"},
    },
    {},
)
```

Kompiliere das Skript zu JavaScript und registriere es über die Flowise-UI. So kann ein Flowise‑Chatbot direkt in Agent‑NN Aufgaben bearbeiten oder Informationen abrufen. Optional lassen sich `method` und `timeout` an den Pluginaufruf übergeben.

## Registrierung der Komponente

1. Wechsle in das Verzeichnis `integrations/flowise-agentnn`.
2. Installiere Abhängigkeiten mit `npm install` und führe `npx tsc` aus.
3. Lade die erzeugte `dist/AgentNN.js` Datei in der Flowise-Administration hoch.
4. Lege beim Einbinden der Komponente die URL deines Agent‑NN Gateways fest und
   optional weitere Parameter wie `taskType`, `method`, zusätzliche HTTP-Header
   oder ein eigenes Timeout.

Weitere Details enthält der [Integration Plan](full_integration_plan.md).
