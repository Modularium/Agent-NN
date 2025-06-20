# Task Dispatcher API

`POST /task`
: Accepts a task with `task_type`, `input` and optional `session_id`.
  The dispatcher looks up the session context, selects a worker via the
  Agent Registry and returns the worker response.

`GET /health`
: Simple health check returning `{"status": "ok"}`.

### Beispiel

Request:
```bash
curl -X POST http://localhost:8000/task \
     -H "Content-Type: application/json" \
     -d '{"task_type": "chat", "input": "Hello"}'
```

Response:
```json
{
    "result": "Hi there!",
    "worker": "worker_dev"
}
```
