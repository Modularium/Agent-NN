import unittest
from unittest.mock import patch, MagicMock
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import tempfile
import shutil
import os
from torch.utils.data import DataLoader, TensorDataset
from nn_models.training_infrastructure import (
    GradientAccumulator,
    ModelCheckpointer,
    DistributedTrainer,
)

pytestmark = pytest.mark.heavy


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestTrainingInfrastructure(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("nn_models.training_infrastructure.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment

        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create test data
        self.batch_size = 4
        self.input_dim = 10

        self.inputs = torch.randn(self.batch_size, self.input_dim)
        self.targets = torch.randn(self.batch_size, 1)

        # Initialize model and optimizer
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Create data loaders
        dataset = TensorDataset(self.inputs, self.targets)
        self.train_loader = DataLoader(dataset, batch_size=2)
        self.val_loader = DataLoader(dataset, batch_size=2)

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_gradient_accumulator(self):
        """Test gradient accumulation."""
        accumulator = GradientAccumulator(self.model, accumulation_steps=2)

        # First step
        loss = torch.tensor(1.0, requires_grad=True)
        accumulator.accumulate(loss)
        self.assertEqual(accumulator.current_step, 1)

        # Second step
        loss = torch.tensor(1.0, requires_grad=True)
        accumulator.accumulate(loss)
        self.assertEqual(accumulator.current_step, 2)

        # Check step
        stepped = accumulator.step()
        self.assertTrue(stepped)
        self.assertEqual(accumulator.current_step, 0)

    def test_model_checkpointer(self):
        """Test model checkpointing."""
        checkpointer = ModelCheckpointer(self.test_dir, "test_model", max_versions=2)

        # Save checkpoints
        metrics = {"loss": 0.5}
        version1 = checkpointer.save(self.model, self.optimizer, metrics)

        metrics = {"loss": 0.3}
        version2 = checkpointer.save(self.model, self.optimizer, metrics)

        metrics = {"loss": 0.4}
        version3 = checkpointer.save(self.model, self.optimizer, metrics)

        # Check versions
        self.assertEqual(len(checkpointer.versions), 2)
        self.assertNotIn(version1, [v["version"] for v in checkpointer.versions])

        # Load checkpoint
        checkpoint = checkpointer.load(version3)
        self.assertIn("model_state", checkpoint)
        self.assertIn("optimizer_state", checkpoint)

        # Get best version
        best_version = checkpointer.get_best_version("loss", higher_better=False)
        self.assertEqual(best_version, version2)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_distributed_trainer(self, mock_rank, mock_cuda):
        """Test distributed trainer."""
        # Initialize trainer
        trainer = DistributedTrainer(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        # Train epoch
        metrics = trainer.train_epoch(epoch=0)
        self.assertIn("loss", metrics)

        # Validate
        metrics = trainer.validate()
        self.assertIn("val_loss", metrics)

        # Full training
        history = trainer.train(num_epochs=2, early_stopping=3)

        # Check history
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertEqual(len(history["train_loss"]), 2)

    def test_batch_preparation(self):
        """Test batch preparation."""
        trainer = DistributedTrainer(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        # Test tensor
        batch = torch.randn(2, 10)
        prepared = trainer._prepare_batch(batch)
        self.assertEqual(prepared.device, trainer.device)

        # Test list
        batch = [torch.randn(2, 10), torch.randn(2, 1)]
        prepared = trainer._prepare_batch(batch)
        self.assertEqual(prepared[0].device, trainer.device)
        self.assertEqual(prepared[1].device, trainer.device)

        # Test dict
        batch = {"input": torch.randn(2, 10), "target": torch.randn(2, 1)}
        prepared = trainer._prepare_batch(batch)
        self.assertEqual(prepared["input"].device, trainer.device)
        self.assertEqual(prepared["target"].device, trainer.device)

    def test_early_stopping(self):
        """Test early stopping."""
        trainer = DistributedTrainer(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        # Train with early stopping
        history = trainer.train(num_epochs=10, early_stopping=2)

        # Check early stopping
        self.assertLess(len(history["train_loss"]), 10)

    def test_mlflow_logging(self):
        """Test MLflow logging."""
        trainer = DistributedTrainer(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        # Train model
        trainer.train(num_epochs=1)

        # Check MLflow calls
        self.mock_mlflow.log_params.assert_called()
        self.mock_mlflow.log_metrics.assert_called()

if __name__ == "__main__":
    unittest.main()
