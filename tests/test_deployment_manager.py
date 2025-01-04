import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil
import os
import yaml
from datetime import datetime
from managers.deployment_manager import DeploymentManager

class TestDeploymentManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config")
        self.compose_path = os.path.join(self.test_dir, "compose")
        
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock Docker
        self.docker_patcher = patch('managers.deployment_manager.docker')
        self.mock_docker = self.docker_patcher.start()
        
        # Set up mock Docker client
        self.mock_docker_client = MagicMock()
        self.mock_docker.from_env.return_value = self.mock_docker_client
        
        # Initialize manager
        self.manager = DeploymentManager(
            config_path=self.config_path,
            docker_compose_path=self.compose_path
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.docker_patcher.stop()
        shutil.rmtree(self.test_dir)
        
    def test_create_deployment(self):
        """Test deployment creation."""
        # Create deployment
        deployment_id = self.manager.create_deployment(
            "test_deployment",
            ["worker1", "worker2"],
            {"api_replicas": 2}
        )
        
        # Check deployment was created
        self.assertIn(deployment_id, self.manager.deployments)
        deployment = self.manager.deployments[deployment_id]
        self.assertEqual(deployment["name"], "test_deployment")
        self.assertEqual(deployment["components"], ["worker1", "worker2"])
        
        # Check compose file was created
        compose_file = os.path.join(
            self.compose_path,
            f"{deployment_id}.yml"
        )
        self.assertTrue(os.path.exists(compose_file))
        
        # Check compose config
        with open(compose_file, "r") as f:
            config = yaml.safe_load(f)
            self.assertIn("redis", config["services"])
            self.assertIn("api", config["services"])
            self.assertIn("worker1", config["services"])
            self.assertIn("worker2", config["services"])
            
    async def test_deploy_undeploy(self):
        """Test deployment and undeployment."""
        # Create deployment
        deployment_id = self.manager.create_deployment(
            "test_deployment",
            ["worker1"],
            {}
        )
        
        # Deploy
        await self.manager.deploy(deployment_id)
        
        # Check status
        deployment = self.manager.deployments[deployment_id]
        self.assertEqual(deployment["status"], "deployed")
        
        # Undeploy
        await self.manager.undeploy(deployment_id)
        
        # Check status
        deployment = self.manager.deployments[deployment_id]
        self.assertEqual(deployment["status"], "removed")
        
    async def test_scale_component(self):
        """Test component scaling."""
        # Create deployment
        deployment_id = self.manager.create_deployment(
            "test_deployment",
            ["worker1"],
            {"worker1_replicas": 1}
        )
        
        # Scale component
        await self.manager.scale_component(
            deployment_id,
            "worker1",
            3
        )
        
        # Check configuration
        deployment = self.manager.deployments[deployment_id]
        self.assertEqual(
            deployment["config"]["worker1_replicas"],
            3
        )
        
    def test_get_deployment_status(self):
        """Test deployment status retrieval."""
        # Create deployment
        deployment_id = self.manager.create_deployment(
            "test_deployment",
            ["worker1"],
            {}
        )
        
        # Mock service status
        mock_service = MagicMock()
        mock_service.name = "worker1"
        mock_service.state = "running"
        mock_service.status = "healthy"
        
        self.mock_docker_client.compose.ps.return_value = [mock_service]
        
        # Get status
        status = self.manager.get_deployment_status(deployment_id)
        
        # Check status
        self.assertEqual(status["name"], "test_deployment")
        self.assertIn("worker1", status["components"])
        self.assertEqual(
            status["components"]["worker1"]["state"],
            "running"
        )
        
    def test_get_deployment_logs(self):
        """Test deployment log retrieval."""
        # Create deployment
        deployment_id = self.manager.create_deployment(
            "test_deployment",
            ["worker1"],
            {}
        )
        
        # Mock service logs
        mock_service = MagicMock()
        mock_service.name = "worker1"
        mock_service.logs.return_value = b"Test logs"
        
        self.mock_docker_client.compose.ps.return_value = [mock_service]
        
        # Get logs
        logs = self.manager.get_deployment_logs(deployment_id)
        
        # Check logs
        self.assertIn("worker1", logs)
        self.assertEqual(logs["worker1"], "Test logs")
        
    def test_get_deployment_metrics(self):
        """Test deployment metrics retrieval."""
        # Create deployment
        deployment_id = self.manager.create_deployment(
            "test_deployment",
            ["worker1"],
            {}
        )
        
        # Mock service stats
        mock_service = MagicMock()
        mock_service.name = "worker1"
        mock_service.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 100},
                "system_cpu_usage": 1000
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 0},
                "system_cpu_usage": 0
            },
            "memory_stats": {
                "usage": 1024*1024*100,  # 100MB
                "limit": 1024*1024*1000  # 1GB
            }
        }
        
        self.mock_docker_client.compose.ps.return_value = [mock_service]
        
        # Get metrics
        metrics = self.manager.get_deployment_metrics(deployment_id)
        
        # Check metrics
        self.assertIn("worker1", metrics)
        self.assertIn("cpu_percent", metrics["worker1"])
        self.assertIn("memory_percent", metrics["worker1"])
        
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid deployment ID
        with self.assertRaises(ValueError):
            await self.manager.deploy("invalid_id")
            
        # Test invalid component
        deployment_id = self.manager.create_deployment(
            "test_deployment",
            ["worker1"],
            {}
        )
        with self.assertRaises(ValueError):
            await self.manager.scale_component(
                deployment_id,
                "invalid_component",
                1
            )
            
        # Test deployment failure
        self.mock_docker_client.compose.up.side_effect = Exception("Deploy failed")
        with self.assertRaises(Exception):
            await self.manager.deploy(deployment_id)
            
        # Check error was recorded
        deployment = self.manager.deployments[deployment_id]
        self.assertEqual(deployment["status"], "failed")
        self.assertEqual(deployment["error"], "Deploy failed")

if __name__ == '__main__':
    unittest.main()