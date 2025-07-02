# n8n Integration

Die Workflow-Plattform [n8n](https://n8n.io/) kann Agent‑NN sowohl aufrufen als auch von Agent‑NN aus gestartet werden. Eigene Nodes erleichtern die Kommunikation.

## Agent‑NN von n8n aus nutzen

### Eigener Node

Ein Beispiel befindet sich unter `integrations/n8n-agentnn/AgentNN.node.ts`. Der Node leitet Aufgaben an den Dispatcher weiter. Der Kern sieht wie folgt aus:

```ts
import { IExecuteFunctions } from 'n8n-core';
import { IDataObject } from 'n8n-workflow';
import axios from 'axios';

export async function execute(this: IExecuteFunctions): Promise<IDataObject[]> {
  const endpoint = this.getNodeParameter('endpoint') as string;
  const taskType = this.getNodeParameter('taskType') as string;
  const payload = this.getNodeParameter('payload') as IDataObject;

  const { data } = await axios.post(`${endpoint}/task`, {
    task_type: taskType,
    input: payload,
  });

  return [data as IDataObject];
}
```

Dieses Skript kann als Custom Node in n8n eingebunden werden und sendet Aufgaben direkt an das Agent‑NN API‑Gateway.

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
