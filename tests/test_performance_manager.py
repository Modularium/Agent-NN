import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import pytest

from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
import json
from datetime import datetime
from managers.performance_manager import PerformanceManager



class TestPerformanceManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("utils.logging_util.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Mock Redis
        self.redis_patcher = patch("managers.performance_manager.aioredis")
        self.mock_redis = self.redis_patcher.start()

        # Set up mock Redis instance
        self.mock_redis_instance = AsyncMock()
        self.mock_redis.from_url.return_value = self.mock_redis_instance

        # Initialize manager
        self.manager = PerformanceManager(max_batch_size=32, max_workers=2)

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.redis_patcher.stop()

    async def test_cache_operations(self):
        """Test cache operations."""
        # Test cache miss
        result = await self.manager.get_cached_result("test_model", {"input": "test"})
        self.assertIsNone(result)

        # Test cache hit
        self.mock_redis_instance.get.return_value = json.dumps({"output": "test"})
        result = await self.manager.get_cached_result("test_model", {"input": "test"})
        self.assertEqual(result["output"], "test")

        # Test cache update
        await self.manager.cache_result(
            "test_model", {"input": "test"}, {"output": "new_test"}
        )
        self.mock_redis_instance.setex.assert_called_once()

    async def test_batch_processing(self):
        """Test batch processing."""
        # Create test inputs
        inputs = [torch.randn(64) for _ in range(10)]

        # Submit to batch processor
        futures = []
        for x in inputs:
            future = await self.manager.add_to_batch("test_model", x)
            futures.append(future)

        # Wait for results
        results = await asyncio.gather(*futures)

        # Check results
        self.assertEqual(len(results), len(inputs))
        for result in results:
            self.assertEqual(result.shape, (64,))

    async def test_worker_selection(self):
        """Test worker selection."""
        # Select workers
        worker1 = self.manager._select_worker()
        self.manager.worker_loads[worker1] = 5

        worker2 = self.manager._select_worker()

        # Check different workers were selected
        self.assertNotEqual(worker1, worker2)

        # Check load balancing
        self.assertEqual(self.manager.worker_loads[worker2], 0)

    async def test_batch_size_optimization(self):
        """Test batch size optimization."""
        # Run optimization
        optimal_size = await self.manager.optimize_batch_size(
            "test_model", min_size=1, max_size=64, num_trials=5
        )

        # Check result
        self.assertGreater(optimal_size, 0)
        self.assertLessEqual(optimal_size, 64)

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Add some test metrics
        self.manager.metrics["inference_time"] = [0.1, 0.2, 0.3]
        self.manager.metrics["batch_size"] = [16, 32, 32]

        # Get metrics
        metrics = self.manager.get_performance_metrics()

        # Check metrics
        self.assertIn("inference_time", metrics)
        self.assertIn("batch_size", metrics)
        self.assertEqual(metrics["inference_time"]["mean"], 0.2)
        self.assertEqual(metrics["batch_size"]["max"], 32)

    async def test_concurrent_batches(self):
        """Test concurrent batch processing."""
        # Create multiple batches
        batch1 = [torch.randn(64) for _ in range(5)]
        batch2 = [torch.randn(64) for _ in range(5)]

        # Submit batches concurrently
        futures1 = []
        futures2 = []

        for x in batch1:
            future = await self.manager.add_to_batch("model1", x)
            futures1.append(future)

        for x in batch2:
            future = await self.manager.add_to_batch("model2", x)
            futures2.append(future)

        # Wait for all results
        results1 = await asyncio.gather(*futures1)
        results2 = await asyncio.gather(*futures2)

        # Check results
        self.assertEqual(len(results1), len(batch1))
        self.assertEqual(len(results2), len(batch2))

    async def test_error_handling(self):
        """Test error handling in batch processing."""

        # Mock error in batch processing
        async def mock_inference(*args):
            raise ValueError("Test error")

        with patch.object(
            self.manager, "_run_batch_inference", side_effect=mock_inference
        ):
            # Submit batch
            with self.assertRaises(ValueError):
                future = await self.manager.add_to_batch("test_model", torch.randn(64))
                await future

    def test_cache_key_generation(self):
        """Test cache key generation."""
        # Test with string input
        key1 = self.manager._generate_cache_key("model1", "test input")
        self.assertIsInstance(key1, str)

        # Test with dict input
        key2 = self.manager._generate_cache_key("model1", {"input": "test"})
        self.assertIsInstance(key2, str)

        # Test deterministic
        key3 = self.manager._generate_cache_key("model1", {"input": "test"})
        self.assertEqual(key2, key3)

if __name__ == "__main__":
    unittest.main()
