# Agent Bus

The Agent Bus is a lightweight in-memory queue that allows agents to exchange
messages during a task run. Each agent can publish messages for another agent and
retrieve its pending messages.

```python
from core import agent_bus

agent_bus.publish("critic", {"type": "feedback", "payload": {"score": 1.0}})
for msg in agent_bus.subscribe("critic"):
    ...
```

Supported message types are `hint`, `vote`, `request`, `feedback` and `broadcast`.
