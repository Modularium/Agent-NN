import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from agents.openhands.base_openhands_agent import OpenHandsAgent
from agents.openhands.docker_agent import DockerAgent
from agents.openhands.compose_agent import DockerComposeAgent

class TestOpenHandsAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock aiohttp session
        self.session_patcher = patch('aiohttp.ClientSession')
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
        self.agent = OpenHandsAgent(
            name="test_agent",
            github_token="test_token"
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.session_patcher.stop()
        
    async def test_submit_code(self):
        """Test code submission."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = self.mock_response
        
        # Submit code
        result = await self.agent.submit_code(
            code="print('test')",
            language="python",
            task_description="Test task"
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
            {"status": "completed"}
        ]
        mock_responses = []
        for resp in responses:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=resp)
            mock_responses.append(mock_response)
            
        self.mock_session_instance.get.return_value.__aenter__.side_effect = mock_responses
        
        # Wait for execution
        result = await self.agent.wait_for_execution(
            "test_id",
            poll_interval=0.1
        )
        
        # Check result
        self.assertEqual(result["status"], "completed")
        
        # Verify API calls
        self.assertEqual(self.mock_session_instance.get.call_count, 3)

class TestDockerAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock aiohttp session
        self.session_patcher = patch('aiohttp.ClientSession')
        self.mock_session = self.session_patcher.start()
        
        # Set up mock session instance
        self.mock_session_instance = AsyncMock()
        self.mock_session.return_value = self.mock_session_instance
        
        # Set up mock response
        self.mock_response = AsyncMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={"status": "success"})
        
        # Initialize agent
        self.agent = DockerAgent(
            name="test_docker",
            github_token="test_token"
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.session_patcher.stop()
        
    async def test_build_image(self):
        """Test Docker image building."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = self.mock_response
        
        # Build image
        result = await self.agent.build_image(
            dockerfile="FROM python:3.9",
            context={"requirements.txt": "requests==2.26.0"},
            tag="test:latest"
        )
        
        # Check result
        self.assertEqual(result["status"], "success")
        
        # Verify API call
        self.mock_session_instance.post.assert_called_once()
        
    async def test_run_container(self):
        """Test container running."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = self.mock_response
        
        # Run container
        result = await self.agent.run_container(
            image="test:latest",
            command="python app.py",
            environment={"DEBUG": "1"}
        )
        
        # Check result
        self.assertEqual(result["status"], "success")
        
        # Verify API call
        self.mock_session_instance.post.assert_called_once()

class TestDockerComposeAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock aiohttp session
        self.session_patcher = patch('aiohttp.ClientSession')
        self.mock_session = self.session_patcher.start()
        
        # Set up mock session instance
        self.mock_session_instance = AsyncMock()
        self.mock_session.return_value = self.mock_session_instance
        
        # Set up mock response
        self.mock_response = AsyncMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={"status": "success"})
        
        # Initialize agent
        self.agent = DockerComposeAgent(
            name="test_compose",
            github_token="test_token"
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.session_patcher.stop()
        
    async def test_deploy_stack(self):
        """Test stack deployment."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = self.mock_response
        
        # Deploy stack
        result = await self.agent.deploy_stack(
            compose_file="version: '3'\\nservices:\\n  web:\\n    image: nginx",
            stack_name="test_stack",
            environment={"DEBUG": "1"}
        )
        
        # Check result
        self.assertEqual(result["status"], "success")
        
        # Verify API call
        self.mock_session_instance.post.assert_called_once()
        
    async def test_scale_service(self):
        """Test service scaling."""
        # Mock response
        self.mock_session_instance.post.return_value.__aenter__.return_value = self.mock_response
        
        # Scale service
        result = await self.agent.scale_service(
            stack_name="test_stack",
            service="web",
            replicas=3
        )
        
        # Check result
        self.assertEqual(result["status"], "success")
        
        # Verify API call
        self.mock_session_instance.post.assert_called_once()

if __name__ == '__main__':
    unittest.main()