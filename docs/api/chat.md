# Chat API

The API gateway exposes simple endpoints for chat interactions.

## POST /chat
Send a message and receive the agent response. A new session is created
automatically if no `session_id` is provided.

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello", "agent": "dev"}'
```

Response:
```json
{
  "session_id": "abc123",
  "worker": "worker_dev",
  "response": {"result": "Hi"},
  "duration": 0.5,
  "confidence": 1.0
}
```

## GET `/chat/history/{session_id}`
Return the stored interaction history for a session.

## POST /chat/feedback
Store user feedback for a specific message.

```json
{
  "session_id": "abc123",
  "index": 0,
  "rating": "good",
  "comment": "useful answer"
}
```
