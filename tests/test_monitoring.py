import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from openhands_api.monitoring import OpenHandsMonitoring

class TestOpenHandsMonitoring(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock Redis
        self.redis_patcher = patch('openhands_api.monitoring.aioredis')
        self.mock_redis = self.redis_patcher.start()
        
        # Mock Docker
        self.docker_patcher = patch('openhands_api.monitoring.docker')
        self.mock_docker = self.docker_patcher.start()
        
        # Mock psutil
        self.psutil_patcher = patch('openhands_api.monitoring.psutil')
        self.mock_psutil = self.psutil_patcher.start()
        
        # Set up mock instances
        self.mock_redis_instance = AsyncMock()
        self.mock_redis.from_url.return_value = self.mock_redis_instance
        
        self.mock_docker_instance = MagicMock()
        self.mock_docker.from_env.return_value = self.mock_docker_instance
        
        # Initialize monitoring
        self.monitoring = OpenHandsMonitoring(
            update_interval=0.1  # Fast updates for testing
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.redis_patcher.stop()
        self.docker_patcher.stop()
        self.psutil_patcher.stop()
        
    async def test_start_stop(self):
        """Test monitoring start and stop."""
        # Start monitoring
        await self.monitoring.start()
        self.assertTrue(self.monitoring.running)
        
        # Stop monitoring
        await self.monitoring.stop()
        self.assertFalse(self.monitoring.running)
        
    async def test_system_metrics(self):
        """Test system metrics collection."""
        # Mock system metrics
        self.mock_psutil.cpu_percent.return_value = 50.0
        self.mock_psutil.virtual_memory.return_value = MagicMock(used=1024*1024*100)
        self.mock_psutil.disk_usage.return_value = MagicMock(used=1024*1024*1000)
        
        # Start monitoring
        await self.monitoring.start()
        
        # Wait for metrics update
        await asyncio.sleep(0.2)
        
        # Check metrics
        self.assertEqual(
            self.monitoring.METRICS["cpu_usage"]._value.get(),
            50.0
        )
        self.assertEqual(
            self.monitoring.METRICS["memory_usage"]._value.get(),
            1024*1024*100
        )
        self.assertEqual(
            self.monitoring.METRICS["disk_usage"]._value.get(),
            1024*1024*1000
        )
        
        await self.monitoring.stop()
        
    async def test_docker_metrics(self):
        """Test Docker metrics collection."""
        # Mock Docker containers
        mock_containers = [
            MagicMock(status="running"),
            MagicMock(status="running"),
            MagicMock(status="exited")
        ]
        self.mock_docker_instance.containers.list.return_value = mock_containers
        
        # Mock Docker images
        mock_images = [MagicMock(), MagicMock()]
        self.mock_docker_instance.images.list.return_value = mock_images
        
        # Start monitoring
        await self.monitoring.start()
        
        # Wait for metrics update
        await asyncio.sleep(0.2)
        
        # Check metrics
        self.assertEqual(
            self.monitoring.METRICS["container_count"].labels(
                status="running"
            )._value.get(),
            2
        )
        self.assertEqual(
            self.monitoring.METRICS["container_count"].labels(
                status="exited"
            )._value.get(),
            1
        )
        self.assertEqual(
            self.monitoring.METRICS["image_count"]._value.get(),
            2
        )
        
        await self.monitoring.stop()
        
    async def test_execution_metrics(self):
        """Test execution metrics collection."""
        # Mock Redis data
        self.mock_redis_instance.keys.return_value = [
            "execution:1",
            "execution:2"
        ]
        self.mock_redis_instance.hgetall.side_effect = [
            {
                "language": "python",
                "status": "completed",
                "created_at": "2024-01-01T12:00:00",
                "completed_at": "2024-01-01T12:00:10"
            },
            {
                "language": "javascript",
                "status": "failed",
                "created_at": "2024-01-01T12:00:00",
                "completed_at": "2024-01-01T12:00:05"
            }
        ]
        
        # Start monitoring
        await self.monitoring.start()
        
        # Wait for metrics update
        await asyncio.sleep(0.2)
        
        # Check metrics
        self.assertEqual(
            self.monitoring.METRICS["executions_total"].labels(
                language="python",
                status="completed"
            )._value.get(),
            1
        )
        self.assertEqual(
            self.monitoring.METRICS["executions_total"].labels(
                language="javascript",
                status="failed"
            )._value.get(),
            1
        )
        
        await self.monitoring.stop()
        
    def test_api_metrics(self):
        """Test API metrics tracking."""
        # Track some requests
        self.monitoring.track_api_request(
            endpoint="/execute",
            method="POST",
            status=200,
            duration=0.5
        )
        self.monitoring.track_api_request(
            endpoint="/execute",
            method="POST",
            status=500,
            duration=1.0
        )
        
        # Check metrics
        self.assertEqual(
            self.monitoring.METRICS["api_requests"].labels(
                endpoint="/execute",
                method="POST",
                status=200
            )._value.get(),
            1
        )
        self.assertEqual(
            self.monitoring.METRICS["api_requests"].labels(
                endpoint="/execute",
                method="POST",
                status=500
            )._value.get(),
            1
        )
        
    async def test_metrics_summary(self):
        """Test metrics summary generation."""
        # Mock some metrics
        self.monitoring.METRICS["cpu_usage"].set(50.0)
        self.monitoring.METRICS["memory_usage"].set(1024*1024*100)
        self.monitoring.METRICS["container_count"].labels(
            status="running"
        ).set(2)
        self.monitoring.METRICS["executions_total"].labels(
            language="python",
            status="completed"
        ).inc()
        
        # Get summary
        summary = await self.monitoring.get_metrics_summary()
        
        # Check summary structure
        self.assertIn("system", summary)
        self.assertIn("docker", summary)
        self.assertIn("executions", summary)
        self.assertIn("api", summary)
        
        # Check values
        self.assertEqual(summary["system"]["cpu_usage"], 50.0)
        self.assertEqual(
            summary["docker"]["containers"]["running"],
            2
        )
        self.assertEqual(summary["executions"]["total"], 1)

if __name__ == '__main__':
    unittest.main()