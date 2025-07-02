# n8n Integration

Die Workflow-Plattform [n8n](https://n8n.io/) kann Agent-NN über REST-Endpunkte ansprechen.
Hierfür steht ein Plugin `n8n_workflow` bereit, das einen beliebigen Webhook oder
REST-Workflow aufruft.

## Beispiel

```python
from plugins.n8n_workflow.plugin import Plugin

plugin = Plugin()
result = plugin.execute({"url": "http://localhost:5678/webhook/test", "payload": {"text": "Hallo"}}, {})
```

Der Aufruf gibt die Antwort des Workflows zurück. In n8n muss dazu ein Webhook-
Node oder eine HTTP Request URL bereitgestellt werden.
