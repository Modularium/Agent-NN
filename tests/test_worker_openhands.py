import asyncio
import json
import os
import sys
import types

import pytest

pytest.importorskip("aiohttp", reason="aiohttp not installed")

module_mlflow = types.ModuleType("mlflow")
setattr(module_mlflow, "active_run", lambda: None)
setattr(module_mlflow, "start_run", lambda *a, **kw: None)
setattr(module_mlflow, "end_run", lambda: None)
sys.modules.setdefault("mlflow", module_mlflow)
sys.modules.setdefault("torch", types.ModuleType("torch"))
setattr(sys.modules["torch"], "Tensor", object)
sys.modules.setdefault("langchain", types.ModuleType("langchain"))
module_schema = types.ModuleType("langchain.schema")
setattr(module_schema, "Document", object)
sys.modules.setdefault("langchain.schema", module_schema)
module_chains = types.ModuleType("langchain.chains")
setattr(module_chains, "RetrievalQA", object)
sys.modules.setdefault("langchain.chains", module_chains)
module_prompts = types.ModuleType("langchain.prompts")
setattr(module_prompts, "PromptTemplate", object)
sys.modules.setdefault("langchain.prompts", module_prompts)
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
module_core_runnables = types.ModuleType("langchain_core.runnables")
setattr(module_core_runnables, "RunnablePassthrough", object)
sys.modules.setdefault("langchain_core.runnables", module_core_runnables)
module_core_llms = types.ModuleType("langchain_core.language_models.llms")
setattr(module_core_llms, "BaseLLM", object)
sys.modules.setdefault("langchain_core.language_models.llms", module_core_llms)
module_core_callbacks = types.ModuleType("langchain_core.callbacks.manager")
setattr(module_core_callbacks, "CallbackManagerForLLMRun", object)
sys.modules.setdefault("langchain_core.callbacks.manager", module_core_callbacks)
sys.modules.setdefault("langchain_openai", types.ModuleType("langchain_openai"))
module_openai = sys.modules["langchain_openai"]
setattr(module_openai, "OpenAI", object)
setattr(module_openai, "ChatOpenAI", object)
setattr(module_openai, "OpenAIEmbeddings", object)
sys.modules.setdefault("datastores.worker_agent_db", types.ModuleType("datastore_db"))


class _DB:
    def __init__(self, name: str):
        self.name = name


sys.modules["datastores.worker_agent_db"].WorkerAgentDB = _DB
sys.modules.setdefault("llm_models.specialized_llm", types.ModuleType("special_llm"))


class _Special:
    def __init__(self, *a, **kw):
        pass


sys.modules["llm_models.specialized_llm"].SpecializedLLM = _Special
sys.modules.setdefault("nn_models.agent_nn_v2", types.ModuleType("agent_nn_v2"))
setattr(sys.modules["nn_models.agent_nn_v2"], "AgentNN", object)
setattr(sys.modules["nn_models.agent_nn_v2"], "TaskMetrics", object)
sys.modules.setdefault(
    "agents.agent_communication", types.ModuleType("agent_communication")
)
setattr(sys.modules["agents.agent_communication"], "AgentCommunicationHub", object)
setattr(sys.modules["agents.agent_communication"], "AgentMessage", object)


class _MT:
    RESPONSE = "response"


setattr(sys.modules["agents.agent_communication"], "MessageType", _MT)
sys.modules.setdefault("agents.domain_knowledge", types.ModuleType("domain_knowledge"))
setattr(sys.modules["agents.domain_knowledge"], "DomainKnowledgeManager", object)

from mcp.worker_openhands.service import WorkerService
from tests.mocks.fake_openhands import FakeOpenHandsServer


class DummyDocker:
    def __init__(self, *a, **kw):
        pass

    async def initialize(self):
        pass

    async def run_container(self, **kwargs):
        return {"container_id": "mock123", "status": "running", "logs": ""}


@pytest.mark.asyncio
async def test_worker_openhands_success(monkeypatch):
    async with FakeOpenHandsServer() as server:
        monkeypatch.setenv("ENABLE_OPENHANDS", "true")
        monkeypatch.setenv("OPENHANDS_API_URL", server.url)
        monkeypatch.setenv("OPENHANDS_JWT", "dummy")
        monkeypatch.setattr("mcp.worker_openhands.service.DockerAgent", DummyDocker)
        service = WorkerService()
        payload = json.dumps({"operation": "start_container", "image": "alpine"})
        result = service.execute_task(payload)
        assert result["status"] == "running"
        assert result["operation_id"] == "mock123"


@pytest.mark.asyncio
async def test_worker_openhands_unauthorized(monkeypatch):
    async with FakeOpenHandsServer() as server:
        monkeypatch.setenv("ENABLE_OPENHANDS", "true")
        monkeypatch.setenv("OPENHANDS_API_URL", server.url)
        monkeypatch.setenv("OPENHANDS_JWT", "dummy")
        monkeypatch.setattr("mcp.worker_openhands.service.DockerAgent", DummyDocker)
        service = WorkerService()
        payload = json.dumps({"operation": "start_container", "fail": "unauthorized"})
        result = service.execute_task(payload)
        assert result["status"] == "error"
        assert "401" in result["error"] or "Unauthorized" in result["error"]


@pytest.mark.asyncio
async def test_worker_openhands_timeout(monkeypatch):
    async with FakeOpenHandsServer() as server:
        monkeypatch.setenv("ENABLE_OPENHANDS", "true")
        monkeypatch.setenv("OPENHANDS_API_URL", server.url)
        monkeypatch.setenv("OPENHANDS_JWT", "dummy")
        monkeypatch.setattr("mcp.worker_openhands.service.DockerAgent", DummyDocker)
        service = WorkerService()

        # monkeypatch run_container to sleep causing timeout
        async def slow_run(*args, **kwargs):
            await asyncio.sleep(31)

        monkeypatch.setattr(service.agent, "run_container", slow_run)

        payload = json.dumps({"operation": "start_container", "image": "alpine"})
        result = service.execute_task(payload)
        assert result["status"] == "error"
        assert result["error"] == "timeout"
