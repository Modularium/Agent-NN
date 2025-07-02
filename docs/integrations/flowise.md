# FlowiseAI Integration

[FlowiseAI](https://flowiseai.com/) bietet eine grafische Oberfläche zum Erstellen von Chatbot‑Flows. Agent‑NN lässt sich sowohl als Quelle für Antworten als auch als externer Aufrufer nutzen.

## Agent‑NN als Flowise Komponente

Unter `integrations/flowise-agentnn` liegt eine Beispielkomponente `AgentNN.ts`. Sie sendet Fragen an den Dispatcher und gibt das Ergebnis zurück. Der Kern sieht so aus:

```ts
import axios from 'axios';

export default class AgentNN {
  constructor(private endpoint: string) {}

  async run(question: string) {
    const { data } = await axios.post(`${this.endpoint}/task`, {
      task_type: 'chat',
      input: question,
    });
    return data.result;
  }
}
```

Diese Komponente wird in Flowise eingebunden und erlaubt es, Benutzeranfragen direkt an Agent‑NN zu delegieren.

## Flowise Workflows aus Agent‑NN anstoßen

Das Plugin `flowise_workflow` ruft HTTP‑basierte Chatflows auf. Es akzeptiert optionale Header und einen Payload:

```python
from plugins.flowise_workflow.plugin import Plugin

plugin = Plugin()
result = plugin.execute(
    {
        "url": "http://localhost:3000/api/v1/predict",
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
4. Lege beim Einbinden der Komponente die URL deines Agent‑NN Gateways fest.

Weitere Details enthält der [Integration Plan](full_integration_plan.md).
