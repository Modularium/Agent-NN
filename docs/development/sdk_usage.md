# Developer SDK

Das SDK ermöglicht einfachen Zugriff auf die Agent-NN-Services.

## Installation

```bash
poetry install
```

## AgentClient Beispiel

```python
from sdk import AgentClient

client = AgentClient()
print(client.submit_task("Hallo Welt"))
```

## ModelManager Beispiel

```python
from sdk import ModelManager

manager = ModelManager()
for m in manager.available_models():
    print(m)
```

Eine Beispielkonfiguration liegt in `~/.agentnnrc`.
