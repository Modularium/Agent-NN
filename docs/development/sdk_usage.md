# Developer SDK

Dieses SDK erleichtert den Zugriff auf Agent‑NN über REST‑Schnittstellen.

## Installation

```bash
pip install -e .[sdk]
```

## Beispiel

```python
from sdk import AgentClient

client = AgentClient()
result = client.submit_task("Erstelle ein Beispiel")
print(result)
```

Eine Beispielkonfiguration in `~/.agentnnrc`:

```json
{
  "host": "http://localhost:8000",
  "api_token": "secret-token"
}
```
