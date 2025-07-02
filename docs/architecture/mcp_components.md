# MCP Components

- **Task-Dispatcher**: central orchestration service. Decides which worker should execute a task.
- **Agent Registry**: stores available worker services, capabilities and health status.
- **Session Manager**: keeps conversation history and temporary state.
- **Vector Store Service**: provides semantic search across documents.
- **LLM Gateway**: exposes a unified API to various LLM backends.
- **User Manager Service**: manages user accounts and tokens.
- **Worker Services**: domain specific executors, e.g. Dev, OpenHands, LOH.
- **API Gateway**: optional entrypoint for external requests with auth and rate limiting.
- **Monitoring/Logging**: collects logs and metrics from all services.

## Service Registration

Worker services announce themselves to the Agent Registry. For local testing the registry loads static data from `mcp/agents.yaml`. Each worker can send a `POST /register` request during startup to appear in the registry. In Phase 1 this process is manual but it prepares automatic discovery for later stages.
