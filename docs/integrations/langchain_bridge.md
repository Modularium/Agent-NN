# LangChain MCP Bridge

`MCPChatWrapper` exposes the MCP server as a LangChain compatible LLM. It wraps
a remote MCP endpoint and forwards prompts as tasks.

```python
from agentnn.integrations.langchain_mcp_adapter import MCPChatWrapper

llm = MCPChatWrapper(endpoint="http://localhost:8090", agent_id="worker_dev")
response = llm.invoke("Explain multi agent systems")
print(response)
```

This wrapper implements the `BaseLLM` interface so it can be used in chains or
agents just like any other LangChain model.
