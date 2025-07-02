# FlowiseAI Integration

[FlowiseAI](https://flowiseai.com/) bietet eine grafische Oberfläche zum Erstellen von Chatbot‑Flows. Agent‑NN lässt sich sowohl als Quelle für Antworten als auch als externer Aufrufer nutzen.

## Agent‑NN als Flowise Komponente

Unter `integrations/flowise-agentnn` liegt eine Beispielkomponente `AgentNN.ts`. Sie sendet Fragen an den Dispatcher und gibt das Ergebnis zurück. Der Kern sieht so aus:

```ts
import axios from 'axios';
import axios, { AxiosRequestConfig } from 'axios';

export default class AgentNN {
  constructor(
    private endpoint: string,
    private taskType = 'chat',
    private headers: Record<string, string> = {},
    private timeout = 10000,
    private method: 'POST' | 'GET' | 'PUT' | 'DELETE' = 'POST',
    private path = '/task',
    private auth?: { username?: string; password?: string; token?: string }
  ) {}

  async run(payload: unknown) {
    const url = `${this.endpoint.replace(/\/$/, '')}${this.path}`;
    const body = { task_type: this.taskType, input: payload };
    const { data } = await axios.request({
    const headers = { ...this.headers } as Record<string, string>;
    if (this.auth?.token) {
      headers['Authorization'] = `Bearer ${this.auth.token}`;
    }
    const opts: AxiosRequestConfig = {
      url,
      method: this.method,
      data: body,
      headers: this.headers,
      headers,
      timeout: this.timeout,
    });
    return data.result ?? data;
    };
    if (this.auth?.username && this.auth?.password) {
      opts.auth = { username: this.auth.username, password: this.auth.password };
    }
    try {
      const { data } = await axios.request(opts);
      return data.result ?? data;
    } catch (err: any) {
      return { error: err.message ?? String(err) };
    }
  }
}
```

Diese Komponente wird in Flowise eingebunden und erlaubt es, Benutzeranfragen direkt an Agent‑NN zu delegieren.

## Quickstart

1. Wechsel in `integrations/flowise-agentnn` und installiere die Abhängigkeiten:
   `npm install && npx tsc`.
2. Lade die erzeugte `dist/AgentNN.js` in der Flowise-UI hoch und setze die
   URL deines Agent‑NN Gateways.
3. Optional können `taskType`, `path`, `method`, `headers`, `auth` und
   `timeout` angepasst werden.

![Flowise Example](flowise_example.png)

Nach dem Upload kann die Komponente sofort verwendet werden. Konfiguriere die
Basis-URL von Agent‑NN sowie optionale Header oder ein anderes Timeout.

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

Kompiliere das Skript zu JavaScript und registriere es über die Flowise-UI. So kann ein Flowise‑Chatbot direkt in Agent‑NN Aufgaben bearbeiten oder Informationen abrufen. Optional lassen sich `path`, `method` und `timeout` an den Pluginaufruf übergeben.

Sobald die Komponente eingebunden ist, kann Agent‑NN automatisiert neue Flowise‑Agenten registrieren und deren Ausführung überwachen. Über das Python‑Plugin werden Aufgabentyp, Modell und Authentifizierungs‑Header an Flowise übermittelt. Das Ergebnis des Flows wird zurück an Agent‑NN gereicht und steht dort für weitere Verarbeitung oder Training zur Verfügung.

## Registrierung der Komponente

1. Wechsle in das Verzeichnis `integrations/flowise-agentnn`.
2. Installiere Abhängigkeiten mit `npm install` und führe `npx tsc` aus.
3. Lade die erzeugte `dist/AgentNN.js` Datei in der Flowise-Administration hoch.
4. Lege beim Einbinden der Komponente die URL deines Agent‑NN Gateways fest und
   optional weitere Parameter wie `taskType`, `path`, `method`, zusätzliche
   HTTP-Header oder ein eigenes Timeout.

Die Flowise-UI nutzt den Plug-in-Manager, um die kompilierten JS-Dateien zu laden.
Stelle deshalb sicher, dass `npm install` und `npx tsc` vor jeder Veröffentlichung
ausgeführt wurden.

Weitere Details enthält der [Integration Plan](full_integration_plan.md).

## Troubleshooting

- **Fehlende Systempakete:** In minimalen Umgebungen kann `npm install` scheitern,
  wenn Build‑Tools wie `make` fehlen. Nutze in diesem Fall den Docker‑Container
  aus dem Repository oder installiere die Pakete manuell.
- **Kein Internetzugang:** Falls Flowise oder Agent‑NN offline laufen, kannst du
  die Plugins mit statischen Antworten testen. Die Python‑Plugins geben bei
  Netzwerkfehlern eine Fehlermeldung zurück, die du im Log findest.
- **Installationsfehler**: Nutze einen internen npm-Mirror, falls `npm install` wegen fehlender Internetverbindung scheitert.
- **Komponente erscheint nicht**: Prüfe, ob die Datei `dist/AgentNN.js` korrekt hochgeladen wurde und Flowise neu gestartet ist.
- **API-Timeouts**: Erhöhe das `timeout` in der Komponente oder teste den Endpoint separat mit `curl`.
