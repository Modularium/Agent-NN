# System Architecture

The MCP version of Agent-NN is composed of several cooperating services. The following diagram illustrates the overall flow.

```mermaid
graph TD
    A[User/CLI] --> B[API Gateway]
    B --> C[Task Dispatcher]
    C --> D[Agent Registry]
    C --> E[Session Manager]
    C --> F[Worker Services]
    C --> G[Vector Store]
    C --> H[LLM Gateway]
    F --> G
    F --> H
```

### Components

- **Task Dispatcher**: central orchestrator that selects the right worker for each task.
- **Agent Registry**: database of all available worker services.
- **Session Manager**: stores ongoing conversation context.
- **Worker Services**: execute domain specific actions.
- **Vector Store**: semantic search across documentation and code.
- **LLM Gateway**: unified access to language models.
- **API Gateway**: optional entry point with authentication and routing.

This setup replaces the former SupervisorAgent architecture and enables independent scaling of each service.
