import unittest
from unittest.mock import patch, MagicMock
import pytest

torch = pytest.importorskip("torch")
import tempfile
import shutil
from torch.utils.data import DataLoader, TensorDataset
from nn_models.multi_task_learning import (
    TaskEncoder,
    AttentionFusion,
    TaskHead,
    MultiTaskNetwork,
    MultiTaskTrainer,
)

pytestmark = pytest.mark.heavy


class TestMultiTaskLearning(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("nn_models.multi_task_learning.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment

        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create test data
        self.batch_size = 4
        self.task_configs = {
            "classification": {"input_dim": 768, "output_dim": 10, "loss": "ce"},
            "regression": {"input_dim": 512, "output_dim": 1, "loss": "mse"},
        }

        # Create test inputs
        self.inputs = {
            "classification": torch.randn(
                self.batch_size, self.task_configs["classification"]["input_dim"]
            ),
            "regression": torch.randn(
                self.batch_size, self.task_configs["regression"]["input_dim"]
            ),
        }

        # Create test targets
        self.targets = {
            "classification": torch.randint(
                0, self.task_configs["classification"]["output_dim"], (self.batch_size,)
            ),
            "regression": torch.randn(
                self.batch_size, self.task_configs["regression"]["output_dim"]
            ),
        }

        # Initialize model
        self.model = MultiTaskNetwork(self.task_configs, shared_dim=256)

        # Initialize trainer
        self.trainer = MultiTaskTrainer(self.model)

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_task_encoder(self):
        """Test task encoder."""
        encoder = TaskEncoder(768, [512, 256], 128)
        x = torch.randn(self.batch_size, 768)
        output = encoder(x)

        self.assertEqual(output.shape, (self.batch_size, 128))

    def test_attention_fusion(self):
        """Test attention fusion."""
        fusion = AttentionFusion(256)
        features = [torch.randn(self.batch_size, 256) for _ in range(3)]
        output = fusion(features)

        self.assertEqual(output.shape, (self.batch_size, 256))

    def test_task_head(self):
        """Test task head."""
        head = TaskHead(256, 128, 10)
        x = torch.randn(self.batch_size, 256)
        output = head(x)

        self.assertEqual(output.shape, (self.batch_size, 10))

    def test_multi_task_network(self):
        """Test multi-task network."""
        # Forward pass
        outputs = self.model(self.inputs)

        # Check outputs
        self.assertIn("classification", outputs)
        self.assertIn("regression", outputs)
        self.assertEqual(
            outputs["classification"].shape,
            (self.batch_size, self.task_configs["classification"]["output_dim"]),
        )
        self.assertEqual(
            outputs["regression"].shape,
            (self.batch_size, self.task_configs["regression"]["output_dim"]),
        )

    def test_loss_computation(self):
        """Test loss computation."""
        # Get model outputs
        outputs = self.model(self.inputs)

        # Compute losses
        losses = self.trainer._compute_loss(outputs, self.targets)

        # Check losses
        self.assertIn("classification", losses)
        self.assertIn("regression", losses)
        self.assertTrue(torch.is_tensor(losses["classification"]))
        self.assertTrue(torch.is_tensor(losses["regression"]))

    def test_training_epoch(self):
        """Test training epoch."""
        # Create dataset and loader
        dataset = TensorDataset(
            torch.stack([self.inputs[task] for task in self.inputs.keys()]),
            torch.stack([self.targets[task] for task in self.targets.keys()]),
        )
        loader = DataLoader(dataset, batch_size=2)

        # Train epoch
        metrics = self.trainer.train_epoch(loader, epoch=0)

        # Check metrics
        self.assertIn("total_loss", metrics)
        self.assertIn("classification_loss", metrics)
        self.assertIn("regression_loss", metrics)

    def test_validation(self):
        """Test validation."""
        # Create dataset and loader
        dataset = TensorDataset(
            torch.stack([self.inputs[task] for task in self.inputs.keys()]),
            torch.stack([self.targets[task] for task in self.targets.keys()]),
        )
        loader = DataLoader(dataset, batch_size=2)

        # Run validation
        metrics = self.trainer.validate(loader)

        # Check metrics
        self.assertIn("val_total_loss", metrics)
        self.assertIn("val_classification_loss", metrics)
        self.assertIn("val_regression_loss", metrics)

    def test_full_training(self):
        """Test full training loop."""
        # Create datasets and loaders
        dataset = TensorDataset(
            torch.stack([self.inputs[task] for task in self.inputs.keys()]),
            torch.stack([self.targets[task] for task in self.targets.keys()]),
        )
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        # Train model
        history = self.trainer.train(
            train_loader, val_loader, num_epochs=2, checkpoint_dir=self.test_dir
        )

        # Check history
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertEqual(len(history["train_loss"]), 2)

        # Check checkpoint
        checkpoint_path = f"{self.test_dir}/best_model.pt"
        self.assertTrue(torch.load(checkpoint_path))

    def test_task_weights(self):
        """Test task weight customization."""
        # Create trainer with custom weights
        weights = {"classification": 0.7, "regression": 0.3}
        trainer = MultiTaskTrainer(self.model, task_weights=weights)

        # Get model outputs
        outputs = self.model(self.inputs)

        # Compute losses
        losses = trainer._compute_loss(outputs, self.targets)

        # Check weighted losses
        self.assertAlmostEqual(
            losses["classification"].item() / losses["regression"].item(),
            weights["classification"] / weights["regression"],
            places=5,
        )

    def test_selective_task_execution(self):
        """Test selective task execution."""
        # Forward pass with selected tasks
        outputs = self.model(self.inputs, tasks=["classification"])

        # Check outputs
        self.assertIn("classification", outputs)
        self.assertNotIn("regression", outputs)

if __name__ == "__main__":
    unittest.main()
