# FlowiseAI Integration

[FlowiseAI](https://flowiseai.com/) ermöglicht das visuelle Erstellen von LLM-Flows.
Mit dem Plugin `flowise_workflow` lassen sich Chatflows per HTTP aus Agent-NN heraus
aufrufen.

## Beispiel

```python
from plugins.flowise_workflow.plugin import Plugin

plugin = Plugin()
result = plugin.execute({"url": "http://localhost:3000/api/v1/predict", "payload": {"question": "Hi"}}, {})
```

So kann ein Flowise-Chatbot direkt in Agent-NN Aufgaben bearbeiten.
