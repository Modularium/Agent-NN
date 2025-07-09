import unittest
import asyncio
import os
import shutil
import tempfile
import pytest

torch = pytest.importorskip("torch")
import numpy as np
from datetime import datetime, timedelta
from managers.system_manager import SystemManager
from managers.cache_manager import CacheManager

pytestmark = pytest.mark.heavy
from managers.model_manager import ModelManager
from managers.knowledge_manager import KnowledgeManager


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for system components."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directories
        cls.test_dir = tempfile.mkdtemp()
        cls.data_dir = os.path.join(cls.test_dir, "data")
        cls.model_dir = os.path.join(cls.test_dir, "models")
        cls.backup_dir = os.path.join(cls.test_dir, "backups")
        cls.config_dir = os.path.join(cls.test_dir, "config")

        # Create directories
        for d in [cls.data_dir, cls.model_dir, cls.backup_dir, cls.config_dir]:
            os.makedirs(d, exist_ok=True)

        # Initialize managers
        cls.system_manager = SystemManager(
            data_dir=cls.data_dir,
            backup_dir=cls.backup_dir,
            config_file=os.path.join(cls.config_dir, "system.json"),
        )

        cls.cache_manager = CacheManager(
            max_size=100, cleanup_interval=1  # 100MB  # 1 second
        )

        cls.model_manager = ModelManager(
            model_dir=cls.model_dir, cache_dir=os.path.join(cls.test_dir, "cache")
        )

        cls.knowledge_manager = KnowledgeManager(data_dir=cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Stop managers
        cls.cache_manager.stop()

        # Remove test directory
        shutil.rmtree(cls.test_dir)

    async def test_system_configuration(self):
        """Test system configuration management."""
        # Update configuration
        config = await self.system_manager.update_config(
            {
                "max_concurrent_tasks": 5,
                "task_timeout": 60,
                "cache_size": 512,
                "log_level": "DEBUG",
            }
        )

        # Verify configuration
        self.assertEqual(config.max_concurrent_tasks, 5)
        self.assertEqual(config.task_timeout, 60)
        self.assertEqual(config.cache_size, 512)
        self.assertEqual(config.log_level, "DEBUG")

        # Get system metrics
        metrics = self.system_manager.get_system_metrics()
        self.assertIn("cpu_percent", metrics)
        self.assertIn("memory_percent", metrics)
        self.assertIn("disk_percent", metrics)

    async def test_backup_restore(self):
        """Test backup and restore functionality."""
        # Create test data
        test_file = os.path.join(self.data_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test data")

        # Create backup
        backup_info = await self.system_manager.create_backup(
            include_models=True, include_data=True
        )

        # Verify backup
        self.assertTrue(os.path.exists(backup_info["path"]))
        self.assertGreater(backup_info["size"], 0)

        # Remove test file
        os.remove(test_file)

        # Restore backup
        await self.system_manager.restore_backup(backup_info["backup_id"])

        # Verify restoration
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as f:
            self.assertEqual(f.read(), "Test data")

    async def test_cache_operations(self):
        """Test cache operations."""
        # Test string cache
        self.cache_manager.set("key1", "value1")
        self.assertEqual(self.cache_manager.get("key1"), "value1")

        # Test tensor cache
        tensor = torch.randn(100, 100)
        self.cache_manager.set("tensor1", tensor)
        cached_tensor = self.cache_manager.get("tensor1")
        self.assertTrue(torch.equal(tensor, cached_tensor))

        # Test embedding cache
        embedding = np.random.randn(100, 100)
        self.cache_manager.set("embedding1", embedding)
        cached_embedding = self.cache_manager.get("embedding1")
        self.assertTrue(np.array_equal(embedding, cached_embedding))

        # Test expiration
        self.cache_manager.set("expire1", "value", ttl=1)
        self.assertEqual(self.cache_manager.get("expire1"), "value")
        await asyncio.sleep(2)
        self.assertIsNone(self.cache_manager.get("expire1"))

        # Test cache stats
        stats = self.cache_manager.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_size", stats)
        self.assertIn("tensor_entries", stats)
        self.assertIn("embedding_entries", stats)

    async def test_model_management(self):
        """Test model management."""
        # Create test model
        model = torch.nn.Linear(10, 1)
        model_path = os.path.join(self.model_dir, "test_model.pt")
        torch.save(model.state_dict(), model_path)

        # Load model
        model_info = await self.model_manager.load_model(
            name="test_model", type="nn", source="local", config={"path": model_path}
        )

        # Verify model
        self.assertIn("model", model_info)
        self.assertEqual(model_info["path"], model_path)

        # Get model versions
        versions = self.model_manager.get_model_versions("test_model")
        self.assertEqual(len(versions), 1)

        # Get model metrics
        metrics = self.model_manager.get_model_metrics("test_model")
        self.assertIsInstance(metrics, dict)

        # Save model version
        version_info = self.model_manager.save_model_version(
            name="test_model", version="v1", metrics={"accuracy": 0.95}
        )
        self.assertEqual(version_info["version"], "v1")

        # Get model versions again
        versions = self.model_manager.get_model_versions("test_model")
        self.assertEqual(len(versions), 2)  # Original + v1

    async def test_knowledge_base(self):
        """Test knowledge base operations."""
        # Create knowledge base
        kb_info = await self.knowledge_manager.create_knowledge_base(
            name="test_kb", domain="test", sources=["test_source"]
        )

        # Verify knowledge base
        self.assertEqual(kb_info["name"], "test_kb")
        self.assertEqual(kb_info["domain"], "test")

        # Add document
        content = b"Test document content"
        doc_id = await self.knowledge_manager.process_document(
            "test_kb", "test.txt", content
        )

        # Search knowledge base
        results = self.knowledge_manager.search_knowledge_base(
            "test_kb", "test", limit=1
        )

        self.assertEqual(len(results), 1)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])

    async def test_system_integration(self):
        """Test system component integration."""
        # End any active MLflow run
        try:
            import mlflow

            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass

        # Update system configuration
        await self.system_manager.update_config(
            {"max_concurrent_tasks": 3, "cache_size": 256}
        )

        # Create and cache model
        model = torch.nn.Linear(10, 1)
        model_path = os.path.join(self.model_dir, "integrated_model.pt")
        torch.save(model.state_dict(), model_path)

        # End any active MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass

        model_info = await self.model_manager.load_model(
            name="integrated_model",
            type="nn",
            source="local",
            config={"path": model_path},
        )

        self.cache_manager.set("model_cache", model_info["model"], ttl=60)

        # Create knowledge base
        kb_info = await self.knowledge_manager.create_knowledge_base(
            name="integrated_kb", domain="test", sources=[]
        )

        # Create backup
        backup_info = await self.system_manager.create_backup()

        # Verify system state
        self.assertIsNotNone(self.cache_manager.get("model_cache"))
        self.assertIn("integrated_model", self.model_manager.list_models())
        self.assertIn(
            "integrated_kb",
            [kb["name"] for kb in self.knowledge_manager.list_knowledge_bases()],
        )

        # Get system metrics
        metrics = self.system_manager.get_system_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("memory_percent", metrics)

        # Get cache stats
        cache_stats = self.cache_manager.get_stats()
        self.assertIsInstance(cache_stats, dict)
        self.assertIn("total_size", cache_stats)

        # Return success
        return True

    async def test_error_handling(self):
        """Test error handling."""
        # Test invalid configuration
        with self.assertRaises(Exception):
            await self.system_manager.update_config({"max_concurrent_tasks": -1})

        # Test invalid backup
        with self.assertRaises(ValueError):
            await self.system_manager.restore_backup("invalid_backup")

        # Test invalid model
        with self.assertRaises(Exception):
            await self.model_manager.load_model(
                name="invalid_model", type="invalid", source="local", config={}
            )

        # Test invalid knowledge base
        with self.assertRaises(ValueError):
            await self.knowledge_manager.process_document(
                "invalid_kb", "test.txt", b"content"
            )


if __name__ == "__main__":    unittest.main()
