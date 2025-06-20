# Agent Registry API

`GET /agents`
: List all registered agents with name, type, URL and capabilities.

`POST /register`
: Register or update an agent. Payload must contain
  `name`, `agent_type`, `url` and `capabilities`.

`GET /agents/health`
: Query the health status of all agents. Each agent's `/health`
  endpoint is called and the status is returned.

### Beispiel

Request:
```bash
curl -X POST http://localhost:8001/register \
     -H "Content-Type: application/json" \
     -d '{"name": "worker_dev", "agent_type": "dev", "url": "http://worker:8101", "capabilities": ["code"]}'
```

Response:
```json
{
    "status": "registered"
}
```
