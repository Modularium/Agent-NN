# Model Context Protocol Integration

Agent-NN exposes a minimal implementation of the **Model Context Protocol (MCP)**
to exchange context information with third party services. The MCP server is
implemented in `agentnn.mcp.mcp_server` and provides the following endpoints:

- `GET /v1/mcp/ping` – health check
- `POST /v1/mcp/execute` – dispatch a `ModelContext` to the internal task
  dispatcher and return the result
- `POST /v1/mcp/task/execute` – alias for `/execute`
- `POST /v1/mcp/agent/create` – register a new worker agent
- `POST /v1/mcp/tool/use` – invoke a plugin tool
- `POST /v1/mcp/context` – store a context in the session manager
- `POST /v1/mcp/context/save` – alias for `/context`
- `GET /v1/mcp/context/{session_id}` – retrieve all contexts for a session
- `GET /v1/mcp/context/get/{session_id}` – alias for `/context/{session_id}`

A lightweight client wrapper is available via `agentnn.mcp.MCPClient`.

Configure the MCP integration using `agentnn/config.yml`:

```yaml
mcp:
  enabled: true
  endpoint: "http://localhost:9000"
  mode: "server"
```

When enabled, external tools can interact with Agent‑NN using the official MCP
schemas. See the official specification for further details.

Example request using ``curl``:

```bash
curl -X POST http://localhost:9000/v1/mcp/execute \
     -H "Content-Type: application/json" \
     -d '{"task_context": {"task_type": "chat", "description": "Hello"}}'
```
