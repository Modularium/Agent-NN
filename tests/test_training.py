"""Tests for training system."""

import os
import unittest
import tempfile
import shutil
from pathlib import Path
import pytest

from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from training.data_logger import (
    InteractionLogger,
    AgentInteractionDataset,
    create_dataloaders,
)
from training.agent_selector_model import AgentSelectorModel, AgentSelectorTrainer



class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test directories
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.log_dir = cls.test_dir / "logs"
        cls.log_dir.mkdir()

        # Create test data
        cls.create_test_data()

    @classmethod
    def create_test_data(cls):
        """Create test interaction data."""
        # Create interaction logs
        interactions_dir = cls.log_dir / "interactions"
        interactions_dir.mkdir()

        # Create sample interactions
        for i in range(10):
            interaction = {
                "task_description": f"Test task {i}",
                "chosen_agent": f"agent_{i % 3}",
                "success": i % 2 == 0,
                "metrics": {
                    "response_time": float(i) / 10,
                    "confidence": 0.8 + float(i) / 100,
                },
                "domain": f"domain_{i % 2}",
                "task_type": f"type_{i % 2}",
            }

            file_path = interactions_dir / f"interaction_{i}.json"
            with open(file_path, "w") as f:
                import json

                json.dump(interaction, f)

    def setUp(self):
        """Set up each test."""
        self.logger = InteractionLogger(str(self.log_dir))

    def test_data_logging(self):
        """Test interaction logging."""
        # Log new interaction
        self.logger.log_interaction(
            task_description="Test task",
            chosen_agent="test_agent",
            success=True,
            metrics={"response_time": 0.5},
        )

        # Verify log file was created
        log_files = list(self.logger.interactions_dir.glob("*.json"))
        self.assertTrue(len(log_files) > 0)

        # Verify log content
        with open(log_files[-1]) as f:
            import json

            data = json.load(f)
            self.assertEqual(data["task_description"], "Test task")
            self.assertEqual(data["chosen_agent"], "test_agent")

    def test_data_preparation(self):
        """Test training data preparation."""
        # Prepare data
        df = self.logger.prepare_training_data(min_interactions=5)

        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("task_embedding", df.columns)
        self.assertIn("agent_encoded", df.columns)

        # Check encoders
        self.assertTrue(hasattr(self.logger.agent_encoder, "classes_"))

    def test_feature_creation(self):
        """Test feature matrix creation."""
        # Prepare data
        df = self.logger.prepare_training_data(min_interactions=5)

        # Create features
        features = self.logger.create_feature_matrix(df)

        # Check tensor
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(len(features), len(df))

        # Create targets
        targets = self.logger.create_target_matrix(df)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual(len(targets), len(df))

    def test_dataset_creation(self):
        """Test dataset creation."""
        # Create dummy data
        features = torch.randn(10, 5)
        targets = torch.randn(10, 3)

        # Create dataset
        dataset = AgentInteractionDataset(features, targets)

        # Test dataset
        self.assertEqual(len(dataset), 10)
        x, y = dataset[0]
        self.assertEqual(x.shape, (5,))
        self.assertEqual(y.shape, (3,))

    def test_dataloader_creation(self):
        """Test dataloader creation."""
        # Create dummy data
        features = torch.randn(100, 5)
        targets = torch.randn(100, 3)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(features, targets, batch_size=32)

        # Check loaders
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)

    def test_model_creation(self):
        """Test model creation and forward pass."""
        # Create model
        model = AgentSelectorModel(
            input_dim=10, hidden_dims=[8, 6], num_agents=3, num_metrics=2
        )

        # Test forward pass
        x = torch.randn(5, 10)
        agent_logits, success_prob, metrics = model(x)

        self.assertEqual(agent_logits.shape, (5, 3))
        self.assertEqual(success_prob.shape, (5, 1))
        self.assertEqual(metrics.shape, (5, 2))

    def test_model_training(self):
        """Test model training."""
        # Create model and trainer
        model = AgentSelectorModel(
            input_dim=10, hidden_dims=[8], num_agents=3, num_metrics=2
        )

        optimizer = torch.optim.Adam(model.parameters())
        trainer = AgentSelectorTrainer(model, optimizer)

        # Create dummy batch
        features = torch.randn(5, 10)
        agent_labels = torch.randint(0, 3, (5,))
        success_labels = torch.randint(0, 2, (5,)).float()
        metric_labels = torch.randn(5, 2)

        # Test training step
        losses = trainer.train_step(
            features, agent_labels, success_labels, metric_labels
        )

        self.assertIn("agent_loss", losses)
        self.assertIn("success_loss", losses)
        self.assertIn("metrics_loss", losses)

    def test_model_validation(self):
        """Test model validation."""
        # Create model and trainer
        model = AgentSelectorModel(
            input_dim=10, hidden_dims=[8], num_agents=3, num_metrics=2
        )

        optimizer = torch.optim.Adam(model.parameters())
        trainer = AgentSelectorTrainer(model, optimizer)

        # Create dummy batch
        features = torch.randn(5, 10)
        agent_labels = torch.randint(0, 3, (5,))
        success_labels = torch.randint(0, 2, (5,)).float()
        metric_labels = torch.randn(5, 2)

        # Test validation step
        metrics = trainer.validate_step(
            features, agent_labels, success_labels, metric_labels
        )

        self.assertIn("agent_accuracy", metrics)
        self.assertIn("success_accuracy", metrics)
        self.assertIn("metrics_mse", metrics)

    def test_model_prediction(self):
        """Test model prediction."""
        # Create model
        model = AgentSelectorModel(
            input_dim=10, hidden_dims=[8], num_agents=3, num_metrics=2
        )

        # Test prediction
        x = torch.randn(1, 10)
        agent_idx, success_prob, metrics = model.predict_agent(x)

        self.assertTrue(agent_idx is None or isinstance(agent_idx, int))
        self.assertIsInstance(success_prob, float)
        self.assertEqual(metrics.shape, (1, 2))

    def test_checkpoint_saving(self):
        """Test model checkpoint saving and loading."""
        # Create model and trainer
        model = AgentSelectorModel(
            input_dim=10, hidden_dims=[8], num_agents=3, num_metrics=2
        )

        optimizer = torch.optim.Adam(model.parameters())
        trainer = AgentSelectorTrainer(model, optimizer)

        # Save checkpoint
        checkpoint_path = self.test_dir / "model.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Load checkpoint
        new_trainer = AgentSelectorTrainer(model, optimizer)
        new_trainer.load_checkpoint(str(checkpoint_path))

        # Verify checkpoint exists
        self.assertTrue(checkpoint_path.exists())

    def tearDown(self):
        """Clean up after each test."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)

if __name__ == "__main__":
    unittest.main()
