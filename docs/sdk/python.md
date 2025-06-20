# Python SDK

Das Python SDK erleichtert die Integration externer Anwendungen mit dem Task-Dispatcher-Service.

## Installation

```bash
pip install path/to/python_agent_nn-*.tar.gz
```

## Beispiel

```python
from agent_nn_client import Client, TaskRequest

client = Client(base_url="http://localhost:8000")
resp = client.create_task_task_post(TaskRequest(task_type="chat", input="Hallo"))
print(resp)
```
