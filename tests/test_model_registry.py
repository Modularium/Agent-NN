import unittest
from unittest.mock import patch, MagicMock
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import tempfile
import shutil
import os
from datetime import datetime
from managers.model_registry import ModelVersion, ModelRegistry

pytestmark = pytest.mark.heavy


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("managers.model_registry.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Set up mock client
        self.mock_client = MagicMock()
        self.mock_mlflow.MlflowClient.return_value = self.mock_client

        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Initialize registry
        self.registry = ModelRegistry(registry_dir=self.test_dir, max_versions=3)

        # Create test model
        self.model = SimpleModel()

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_model_version(self):
        """Test model version."""
        # Create version
        version = ModelVersion(
            version_id="v1",
            model_id="test",
            path="/path/to/model",
            metrics={"loss": 0.5},
            metadata={"epoch": 10},
            timestamp=datetime.now().isoformat(),
        )

        # Test conversion
        data = version.to_dict()
        new_version = ModelVersion.from_dict(data)

        self.assertEqual(version.version_id, new_version.version_id)
        self.assertEqual(version.metrics, new_version.metrics)

    def test_register_model(self):
        """Test model registration."""
        # Register model
        metrics = {"loss": 0.5, "accuracy": 0.95}
        metadata = {"epoch": 10, "batch_size": 32}

        version = self.registry.register_model(
            self.model, "test_model", metrics, metadata
        )

        # Check version
        self.assertIsNotNone(version)
        self.assertEqual(version.metrics, metrics)
        self.assertEqual(version.metadata, metadata)

        # Check file exists
        self.assertTrue(os.path.exists(version.path))

    def test_version_limit(self):
        """Test version limit."""
        # Register multiple versions
        for i in range(5):
            self.registry.register_model(
                self.model, "test_model", {"loss": 0.5 - i * 0.1}
            )

        # Check version count
        versions = self.registry.versions.get("test_model", [])
        self.assertEqual(len(versions), 3)  # max_versions

        # Check oldest version removed
        losses = [v.metrics["loss"] for v in versions]
        self.assertNotIn(0.5, losses)  # First version removed

    def test_get_version(self):
        """Test version retrieval."""
        # Register model
        version = self.registry.register_model(self.model, "test_model", {"loss": 0.5})

        # Get version
        retrieved = self.registry.get_version("test_model", version.version_id)

        self.assertEqual(version.version_id, retrieved.version_id)

        # Get latest
        latest = self.registry.get_version("test_model")
        self.assertEqual(version.version_id, latest.version_id)

    def test_load_model(self):
        """Test model loading."""
        # Register model
        version = self.registry.register_model(self.model, "test_model", {"loss": 0.5})

        # Load model
        loaded = self.registry.load_model("test_model", model_class=SimpleModel)

        self.assertIsInstance(loaded, SimpleModel)

        # Compare parameters
        for p1, p2 in zip(self.model.parameters(), loaded.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_best_version(self):
        """Test best version selection."""
        # Register multiple versions
        metrics = [
            {"loss": 0.5, "accuracy": 0.90},
            {"loss": 0.3, "accuracy": 0.95},
            {"loss": 0.4, "accuracy": 0.92},
        ]

        for m in metrics:
            self.registry.register_model(self.model, "test_model", m)

        # Get best version (loss)
        best_loss = self.registry.get_best_version(
            "test_model", "loss", higher_better=False
        )
        self.assertEqual(best_loss.metrics["loss"], 0.3)

        # Get best version (accuracy)
        best_acc = self.registry.get_best_version(
            "test_model", "accuracy", higher_better=True
        )
        self.assertEqual(best_acc.metrics["accuracy"], 0.95)

    def test_version_comparison(self):
        """Test version comparison."""
        # Register multiple versions
        versions = []
        for i in range(3):
            version = self.registry.register_model(
                self.model, "test_model", {"loss": 0.5 - i * 0.1}
            )
            versions.append(version)

        # Compare versions
        comparison = self.registry.compare_versions(
            "test_model", [v.version_id for v in versions], ["loss"]
        )

        self.assertEqual(len(comparison), 3)
        self.assertTrue("loss" in comparison.columns)

    def test_delete_version(self):
        """Test version deletion."""
        # Register model
        version = self.registry.register_model(self.model, "test_model", {"loss": 0.5})

        # Delete version
        deleted = self.registry.delete_version("test_model", version.version_id)

        self.assertTrue(deleted)
        self.assertFalse(os.path.exists(version.path))

    def test_registry_stats(self):
        """Test registry statistics."""
        # Register multiple models
        for model_id in ["model1", "model2"]:
            for _ in range(2):
                self.registry.register_model(self.model, model_id, {"loss": 0.5})

        # Get stats
        stats = self.registry.get_registry_stats()

        self.assertEqual(stats["total_models"], 2)
        self.assertEqual(stats["total_versions"], 4)
        self.assertIn("model1", stats["models"])
        self.assertIn("model2", stats["models"])

    def test_error_handling(self):
        """Test error handling."""
        # Test invalid model ID
        version = self.registry.get_version("invalid_model")
        self.assertIsNone(version)

        # Test invalid version ID
        version = self.registry.get_version("test_model", "invalid_version")
        self.assertIsNone(version)

        # Test loading invalid model
        model = self.registry.load_model("invalid_model")
        self.assertIsNone(model)

        # Test deleting invalid version
        deleted = self.registry.delete_version("test_model", "invalid_version")
        self.assertFalse(deleted)

if __name__ == "__main__":
    unittest.main()
