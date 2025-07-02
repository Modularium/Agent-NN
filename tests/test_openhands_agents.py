import unittest
from unittest.mock import patch, AsyncMock
import asyncio
import sys
import types

module_mlflow = types.ModuleType("mlflow")
setattr(module_mlflow, "active_run", lambda: None)
setattr(module_mlflow, "start_run", lambda *a, **kw: None)
setattr(module_mlflow, "end_run", lambda: None)
sys.modules.setdefault("mlflow", module_mlflow)
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
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
sys.modules.setdefault(
    "langchain_core.callbacks.manager",
    module_core_callbacks,
)
sys.modules.setdefault("langchain_openai", types.ModuleType("langchain_openai"))
module_openai = sys.modules["langchain_openai"]
setattr(module_openai, "OpenAI", object)
setattr(module_openai, "ChatOpenAI", object)
setattr(module_openai, "OpenAIEmbeddings", object)
sys.modules.setdefault("langchain_core.outputs", types.ModuleType("lc_out"))
setattr(sys.modules["langchain_core.outputs"], "LLMResult", object)
setattr(sys.modules["langchain_core.outputs"], "Generation", object)

from agents.openhands.base_openhands_agent import OpenHandsAgent
from agents.openhands.docker_agent import DockerAgent
from agents.openhands.compose_agent import DockerComposeAgent


class TestOpenHandsAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("utils.logging_util.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Mock aiohttp session
        self.session_patcher = patch("aiohttp.ClientSession")
        self.mock_session = self.session_patcher.start()

        # Set up mock session instance
        self.mock_session_instance = AsyncMock()
        self.mock_session.return_value = self.mock_session_instance

        # Set up mock response
        self.mock_response = AsyncMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={"status": "success"})
        self.mock_response.text = AsyncMock(return_value="success")

        # Initialize agent
        self.agent = OpenHandsAgent(name="test_agent", github_token="test_token")

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.session_patcher.stop()

    async def test_submit_code(self):
        """Test code submission."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = (
            self.mock_response
        )

        # Submit code
        result = await self.agent.submit_code(
            code="print('test')", language="python", task_description="Test task"
        )

        # Check result
        self.assertEqual(result["status"], "success")

        # Verify API call
        self.mock_session_instance.post.assert_called_once()

    async def test_wait_for_execution(self):
        """Test execution waiting."""
        # Mock responses
        responses = [
            {"status": "running"},
            {"status": "running"},
            {"status": "completed"},
        ]
        mock_responses = []
        for resp in responses:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=resp)
            mock_responses.append(mock_response)

        self.mock_session_instance.get.return_value.__aenter__.side_effect = (
            mock_responses
        )

        # Wait for execution
        result = await self.agent.wait_for_execution("test_id", poll_interval=0.1)

        # Check result
        self.assertEqual(result["status"], "completed")

        # Verify API calls
        self.assertEqual(self.mock_session_instance.get.call_count, 3)


class TestDockerAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("utils.logging_util.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Mock aiohttp session
        self.session_patcher = patch("aiohttp.ClientSession")
        self.mock_session = self.session_patcher.start()

        # Set up mock session instance
        self.mock_session_instance = AsyncMock()
        self.mock_session.return_value = self.mock_session_instance

        # Set up mock response
        self.mock_response = AsyncMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={"status": "success"})

        # Initialize agent
        self.agent = DockerAgent(name="test_docker", github_token="test_token")

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.session_patcher.stop()

    async def test_build_image(self):
        """Test Docker image building."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = (
            self.mock_response
        )

        # Build image
        result = await self.agent.build_image(
            dockerfile="FROM python:3.9",
            context={"requirements.txt": "requests==2.26.0"},
            tag="test:latest",
        )

        # Check result
        self.assertEqual(result["status"], "success")

        # Verify API call
        self.mock_session_instance.post.assert_called_once()

    async def test_run_container(self):
        """Test container running."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = (
            self.mock_response
        )

        # Run container
        result = await self.agent.run_container(
            image="test:latest", command="python app.py", environment={"DEBUG": "1"}
        )

        # Check result
        self.assertEqual(result["status"], "success")

        # Verify API call
        self.mock_session_instance.post.assert_called_once()


class TestDockerComposeAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("utils.logging_util.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Mock aiohttp session
        self.session_patcher = patch("aiohttp.ClientSession")
        self.mock_session = self.session_patcher.start()

        # Set up mock session instance
        self.mock_session_instance = AsyncMock()
        self.mock_session.return_value = self.mock_session_instance

        # Set up mock response
        self.mock_response = AsyncMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={"status": "success"})

        # Initialize agent
        self.agent = DockerComposeAgent(name="test_compose", github_token="test_token")

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.session_patcher.stop()

    async def test_deploy_stack(self):
        """Test stack deployment."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = (
            self.mock_response
        )

        # Deploy stack
        result = await self.agent.deploy_stack(
            compose_file="version: '3'\\nservices:\\n  web:\\n    image: nginx",
            stack_name="test_stack",
            environment={"DEBUG": "1"},
        )

        # Check result
        self.assertEqual(result["status"], "success")

        # Verify API call
        self.mock_session_instance.post.assert_called_once()

    async def test_scale_service(self):
        """Test service scaling."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = (
            self.mock_response
        )

        # Scale service
        result = await self.agent.scale_service(
            stack_name="test_stack", service="web", replicas=3
        )

        # Check result
        self.assertEqual(result["status"], "success")

        # Verify API call
        self.mock_session_instance.post.assert_called_once()


class TestConversationEndpoints(unittest.TestCase):
    def setUp(self):
        self.mlflow_patcher = patch("utils.logging_util.mlflow")
        self.mlflow_patcher.start()
        self.session_patcher = patch("aiohttp.ClientSession")
        self.mock_session = self.session_patcher.start()

        self.mock_session_instance = AsyncMock()
        self.mock_session.return_value = self.mock_session_instance

        self.mock_response = AsyncMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={"conversation_id": "c1"})
        self.mock_response.text = AsyncMock(return_value="ok")

        self.agent = OpenHandsAgent(name="conv_agent", github_token="token")

    def tearDown(self):
        self.mlflow_patcher.stop()
        self.session_patcher.stop()

    async def test_start_conversation(self):
        self.mock_session_instance.post.return_value.__aenter__.return_value = (
            self.mock_response
        )
        result = await self.agent.start_conversation("hello")
        self.assertEqual(result["conversation_id"], "c1")
        self.mock_session_instance.post.assert_called_once()

    async def test_get_trajectory(self):
        self.mock_response.json = AsyncMock(return_value={"trajectory": ["hi"]})
        self.mock_session_instance.get.return_value.__aenter__.return_value = (
            self.mock_response
        )
        result = await self.agent.get_trajectory("c1")
        self.assertEqual(result["trajectory"], ["hi"])
        self.mock_session_instance.get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
