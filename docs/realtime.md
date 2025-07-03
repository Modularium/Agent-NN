# Real-time Session Events

`agentnn.mcp.mcp_ws` exposes a WebSocket endpoint for live updates. Connect to
`/ws/session/<id>` to receive JSON events about session creation, added agents
and agent results.

Example event structure:

```json
{"event": "agent_result", "session_id": "abc", "agent": "a1", "result": "done"}
```
