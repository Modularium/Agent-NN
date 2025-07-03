"""LangChain wrapper for the MCP server."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

from core.model_context import ModelContext, TaskContext
from ..mcp.mcp_client import MCPClient


class MCPChatWrapper(BaseLLM):
    """LangChain-compatible LLM wrapper that delegates to an MCP server."""

    endpoint: str
    agent_id: str = "default"

    def __init__(self, endpoint: str = "http://localhost:8090", agent_id: str = "default", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = MCPClient(endpoint)
        self.endpoint = endpoint
        self.agent_id = agent_id

    @property
    def _llm_type(self) -> str:
        return "mcp-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        ctx = ModelContext(
            task_context=TaskContext(task_type="chat", description=prompt),
            agent_selection=self.agent_id,
        )
        result = self.client.execute(ctx)
        return result.result or ""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = [
            [Generation(text=self._call(p, stop=stop, run_manager=run_manager, **kwargs))]
            for p in prompts
        ]
        return LLMResult(generations=generations)
