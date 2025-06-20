# Agent Registry API

`GET /agents`
: List all registered agents with name, type, URL and capabilities.

`POST /register`
: Register or update an agent. Payload must contain
  `name`, `agent_type`, `url` and `capabilities`.

`GET /agents/health`
: Query the health status of all agents. Each agent's `/health`
  endpoint is called and the status is returned.
