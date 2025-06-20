# MCP Components

- **Task-Dispatcher**: central orchestration service. Decides which worker should execute a task.
- **Agent Registry**: stores available worker services, capabilities and health status.
- **Session Manager**: keeps conversation history and temporary state.
- **Vector Store Service**: provides semantic search across documents.
- **LLM Gateway**: exposes a unified API to various LLM backends.
- **Worker Services**: domain specific executors, e.g. Dev, OpenHands, LOH.
- **API Gateway**: optional entrypoint for external requests with auth and rate limiting.
- **Monitoring/Logging**: collects logs and metrics from all services.
