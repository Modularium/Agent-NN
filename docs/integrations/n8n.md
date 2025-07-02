# n8n Integration

Die Workflow-Plattform [n8n](https://n8n.io/) kann Agent‑NN sowohl aufrufen als auch von Agent‑NN aus gestartet werden. Eigene Nodes erleichtern die Kommunikation.

## Agent‑NN von n8n aus nutzen

### Eigener Node

Ein Beispiel befindet sich unter `integrations/n8n-agentnn/AgentNN.node.ts`. Der Node leitet Aufgaben an den Dispatcher weiter. Der Kern sieht wie folgt aus:

```ts
import { IExecuteFunctions } from 'n8n-core';
import { IDataObject } from 'n8n-workflow';
import axios, { AxiosRequestConfig } from 'axios';

export async function execute(this: IExecuteFunctions): Promise<IDataObject[]> {
  const endpoint = this.getNodeParameter('endpoint') as string;
  const taskType = this.getNodeParameter('taskType') as string;
  const payload = this.getNodeParameter('payload') as IDataObject;
  const method = (this.getNodeParameter('method', 0, 'POST') as string).toUpperCase();
  const headers = (this.getNodeParameter('headers', 0, {}) as IDataObject) as Record<string, string>;
  const timeout = this.getNodeParameter('timeout', 0, 10000) as number;

  const options: AxiosRequestConfig = {
    method,
    url: `${endpoint}/task`,
    data: {
      task_type: taskType,
      input: payload,
    },
    headers,
    timeout,
  };

  const { data } = await axios.request(options);

  return [data as IDataObject];
}
```

Dieses Skript kann als Custom Node in n8n eingebunden werden und sendet Aufgaben direkt an das Agent‑NN API‑Gateway.
Dabei können optionale Parameter wie `method`, `headers` und `timeout` gesetzt werden, um HTTP-Methode, Header und Zeitlimit zu steuern.

## Agent‑NN ruft n8n Workflows auf

Das Python‑Plugin `n8n_workflow` erlaubt es, beliebige Webhook‑ oder REST‑Workflows anzusprechen. Es unterstützt optionale Header und Payloads:

```python
from plugins.n8n_workflow.plugin import Plugin

plugin = Plugin()
result = plugin.execute(
    {
        "url": "http://localhost:5678/webhook/test",
        "payload": {"text": "Hallo"},
        "headers": {"X-Api-Key": "token"},
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
4. Starte n8n neu, um den Node nutzen zu können.

Weitere Hinweise zur Konfiguration findest du im [Integration Plan](full_integration_plan.md).
