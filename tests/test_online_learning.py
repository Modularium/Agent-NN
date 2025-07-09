import unittest
from unittest.mock import patch, MagicMock
import pytest

from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
import torch.nn as nn
import tempfile
import threading
import queue
import time
from nn_models.online_learning import (
    StreamingBuffer,
    StreamingDataset,
    AdaptiveLearningRate,
    OnlineLearner,
)



class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 10)

    def forward(self, batch):
        x = batch["input"]
        y = batch["target"]
        out = self.linear(x)
        loss = nn.functional.mse_loss(out, y)
        return loss


class TestOnlineLearning(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("nn_models.online_learning.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment

        # Create test data
        self.batch_size = 4
        self.input_dim = 768
        self.output_dim = 10

        # Initialize model
        self.model = SimpleModel()

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()

    def test_streaming_buffer(self):
        """Test streaming buffer."""
        buffer = StreamingBuffer(
            capacity=10,
            feature_dims={"input": self.input_dim, "target": self.output_dim},
        )

        # Add data
        data = {
            "input": torch.randn(self.input_dim),
            "target": torch.randn(self.output_dim),
        }

        added = buffer.add(data)
        self.assertTrue(added)
        self.assertEqual(buffer.count, 1)

        # Get batch
        batch = buffer.get_batch(batch_size=2)
        self.assertIsNotNone(batch)
        self.assertEqual(batch["input"].shape[1], self.input_dim)
        self.assertEqual(batch["target"].shape[1], self.output_dim)

    def test_streaming_dataset(self):
        """Test streaming dataset."""
        data_queue = queue.Queue()
        dataset = StreamingDataset(data_queue)

        # Add data
        data = {
            "input": torch.randn(self.input_dim),
            "target": torch.randn(self.output_dim),
        }
        data_queue.put(data)

        # Get data
        iterator = iter(dataset)
        result = next(iterator)

        self.assertEqual(result["input"].shape, data["input"].shape)

    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate."""
        adaptive_lr = AdaptiveLearningRate(init_lr=0.001)

        # Test increase
        lr = adaptive_lr.update(1.0)
        lr = adaptive_lr.update(0.8)  # Decreasing loss
        self.assertGreater(lr, 0.001)

        # Test decrease
        lr = adaptive_lr.update(1.2)  # Increasing loss
        self.assertLess(lr, 0.001 * 1.05)

    def test_online_learner(self):
        """Test online learner."""
        learner = OnlineLearner(
            self.model, buffer_capacity=100, batch_size=4, update_interval=0.1
        )

        # Start learning
        learner.start()

        # Add data
        for _ in range(10):
            data = {
                "input": torch.randn(self.input_dim),
                "target": torch.randn(self.output_dim),
            }
            learner.add_data(data)

        # Wait for updates
        time.sleep(0.5)

        # Check stats
        stats = learner.get_stats()
        self.assertGreater(stats["buffer_size"], 0)
        self.assertGreater(stats["updates"], 0)

        # Stop learning
        learner.stop()

    def test_state_save_load(self):
        """Test state saving and loading."""
        learner = OnlineLearner(self.model)

        # Create temporary file
        with tempfile.NamedTemporaryFile() as tmp:
            # Save state
            learner.save_state(tmp.name)

            # Create new learner
            new_learner = OnlineLearner(SimpleModel())

            # Load state
            new_learner.load_state(tmp.name)

            # Check states match
            self.assertEqual(learner.update_count, new_learner.update_count)

    def test_error_handling(self):
        """Test error handling."""
        learner = OnlineLearner(self.model)

        # Test invalid data
        invalid_data = {
            "input": torch.randn(10),  # Wrong dimension
            "target": torch.randn(self.output_dim),
        }

        # This should not crash
        learner.add_data(invalid_data)

        # Start and stop
        learner.start()
        time.sleep(0.1)
        learner.stop()

    def test_concurrent_updates(self):
        """Test concurrent updates."""
        learner = OnlineLearner(self.model, update_interval=0.1)

        # Start learning
        learner.start()

        # Create producer thread
        def produce_data():
            for _ in range(20):
                data = {
                    "input": torch.randn(self.input_dim),
                    "target": torch.randn(self.output_dim),
                }
                learner.add_data(data)
                time.sleep(0.05)

        producer = threading.Thread(target=produce_data)
        producer.start()

        # Wait for producer
        producer.join()

        # Wait for processing
        time.sleep(0.5)

        # Stop learning
        learner.stop()

        # Check updates occurred
        self.assertGreater(learner.update_count, 0)

    def test_model_improvement(self):
        """Test model improvement over time."""
        learner = OnlineLearner(self.model, update_interval=0.1)

        # Start learning
        learner.start()

        # Track initial loss
        initial_loss = None

        # Generate consistent training data
        input_data = torch.randn(self.input_dim)
        target_data = torch.randn(self.output_dim)

        # Add training data
        for _ in range(50):
            data = {
                "input": input_data,  # Use same input
                "target": target_data,  # Use same target
            }
            learner.add_data(data)

            # Record initial loss
            if initial_loss is None and learner.adaptive_lr.loss_history:
                initial_loss = learner.adaptive_lr.loss_history[0]

            time.sleep(0.02)

        # Wait for processing
        time.sleep(1.0)  # Wait longer

        # Stop learning
        learner.stop()

        # Check loss decreased
        if initial_loss is not None:
            final_loss = learner.adaptive_lr.loss_history[-1]
            self.assertLess(final_loss, initial_loss)

if __name__ == "__main__":
    unittest.main()
