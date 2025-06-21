# Multi-Agent Execution

Phase 2 introduces a coordination layer that allows several worker agents to collaborate on a single task. The dispatcher now supports three modes:

- **single** – legacy behaviour; exactly one agent handles the task
- **parallel** – all matching agents receive the task simultaneously
- **orchestrated** – predefined pipeline of roles (retriever → summarizer → writer)

The `AgentCoordinator` service receives a `ModelContext` with the selected agents and executes them according to the chosen mode. Results of each agent run are stored in `ModelContext.agents` and the final `aggregated_result` is returned to the dispatcher.
