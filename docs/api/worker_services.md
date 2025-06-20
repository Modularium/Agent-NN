# Worker Services API

Each worker exposes `/execute_task` to perform a domain specific action via the LLM Gateway.

- **worker_dev** – Generates small Python code snippets.
- **worker_loh** – Provides short caregiving answers.
- **worker_openhands** – Mocks Docker container operations.

All services also implement `/health`.


### Beispiel

Request:
```bash
curl -X POST http://localhost:8101/execute_task \
     -H "Content-Type: application/json" \
     -d '{"input": "print(1+1)"}'
```

Response:
```json
{
    "output": "2"
}
```
