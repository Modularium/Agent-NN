# n8n Integration

Die Workflow-Plattform [n8n](https://n8n.io/) kann Agent‑NN sowohl aufrufen als auch von Agent‑NN aus gestartet werden. Eigene Nodes erleichtern die Kommunikation.

## Agent‑NN von n8n aus nutzen

### Eigener Node

Ein Beispiel befindet sich unter `integrations/n8n-agentnn/AgentNN.node.ts`. Der Node leitet Aufgaben an den Dispatcher weiter. Der Kern sieht wie folgt aus:

```ts
import {
  IExecuteFunctions,
  IDataObject,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  NodeConnectionType,
} from 'n8n-workflow';
import axios, { AxiosRequestConfig } from 'axios';

export class AgentNN implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'AgentNN',
    name: 'agentnn',
    group: ['transform'],
    version: 1,
    inputs: [NodeConnectionType.Main],
    outputs: [NodeConnectionType.Main],
    properties: [
      { displayName: 'Endpoint', name: 'endpoint', type: 'string', default: 'http://localhost:8000' },
      { displayName: 'Task Type', name: 'taskType', type: 'string', default: 'chat' },
      { displayName: 'Payload', name: 'payload', type: 'json', default: '{}' },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const endpoint = this.getNodeParameter('endpoint', 0) as unknown as string;
    const taskType = this.getNodeParameter('taskType', 0) as unknown as string;
    const payload = this.getNodeParameter('payload', 0) as unknown as IDataObject;

    const options: AxiosRequestConfig = {
      method: 'POST',
      url: `${endpoint}/task`,
      data: { task_type: taskType, input: payload },
    };

    const { data } = await axios.request(options);
    return [[data as INodeExecutionData]];
  }
}
```

Dieses Skript kann als Custom Node in n8n eingebunden werden und sendet Aufgaben direkt an das Agent‑NN API‑Gateway.
Dabei können optionale Parameter wie `path`, `method`, `headers`, `timeout` und `auth` gesetzt werden, um URL-Pfad, HTTP-Methode, Header, Zeitlimit und Basis-Auth zu steuern.

## Quick Start

Schneller Einstieg zum Testen des Nodes:

```bash
cd integrations/n8n-agentnn
npm install
npx tsc
cp dist/* ~/.n8n/custom/
n8n start
```

Nach dem Kopieren der Dateien erscheint der Node im Editor unter `AgentNN`.

## Agent‑NN ruft n8n Workflows auf

Das Python‑Plugin `n8n_workflow` erlaubt es, beliebige Webhook‑ oder REST‑Workflows anzusprechen. Neben Headern und Payloads lassen sich auch die HTTP-Methode, ein Timeout und optionale Basic‑Auth-Daten definieren:


```python
from plugins.n8n_workflow.plugin import Plugin

plugin = Plugin()
result = plugin.execute(
    {
        "endpoint": "http://localhost:5678",
        "path": "/webhook/test",
        "payload": {"text": "Hallo"},
        "headers": {"X-Api-Key": "token"},
        "auth": {"username": "user", "password": "pass"},
    },
    {},
)
```

Der Aufruf gibt die Antwort des Workflows zurück.

Über die optionalen Parameter `method` und `timeout` lassen sich HTTP-Methode und Zeitlimit anpassen.

## Installation des Nodes

1. Navigiere in das Verzeichnis `integrations/n8n-agentnn`.
2. Führe `npm install` aus und kompiliere den TypeScript-Code mit `npx tsc`.
3. Kopiere die erzeugten Dateien aus `dist/` in den `~/.n8n/custom` Ordner deiner n8n-Installation.
4. Starte n8n neu, damit der PluginManager das kompilerte JavaScript laden kann. In den Node-Einstellungen lassen sich `taskType`, `path`, `method`, Header und Timeout konfigurieren.

Vor einer Veröffentlichung muss der Node immer nach `dist/` kompiliert werden. Der PluginManager lädt ausschließlich die JavaScript-Dateien.

Weitere Hinweise zur Konfiguration findest du im [Integration Plan](full_integration_plan.md).

## Troubleshooting

- **Fehlende Module**: Schlägt `npm install` fehl, prüfe die Proxy-Einstellungen oder verwende ein internes npm-Registry.
- **Keine Ausführung**: Wird der Node nicht angezeigt, kontrolliere, ob die Dateien unter `~/.n8n/custom` die Endung `.js` besitzen und n8n neu gestartet wurde.
- **Netzwerkprobleme**: Bei Verbindungsfehlern setze das Feld `timeout` höher oder teste die URL per `curl`.
- **Offline-Umgebungen**: Nutze einen lokalen npm-Cache (z.B. `npm config set cache /path/to/cache`) oder spiegel die Pakete intern.
