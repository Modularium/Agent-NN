import unittest
from unittest.mock import patch, MagicMock
import torch
import tempfile
import shutil
from torch.utils.data import DataLoader
from nn_models.advanced_training import (
    MultiModalDataset,
    HierarchicalNetwork,
    AdvancedTrainer
)

class TestAdvancedTraining(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('nn_models.advanced_training.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment
        
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.batch_size = 4
        self.text_dim = 768
        self.metric_dim = 10
        self.feedback_dim = 5
        
        self.text_embeddings = torch.randn(
            self.batch_size,
            self.text_dim
        )
        self.metrics = torch.randn(
            self.batch_size,
            self.metric_dim
        )
        self.feedback = torch.randn(
            self.batch_size,
            self.feedback_dim
        )
        
        # Initialize model
        self.model = HierarchicalNetwork(
            text_dim=self.text_dim,
            metric_dim=self.metric_dim,
            feedback_dim=self.feedback_dim,
            hidden_dims=[512, 256, 128, 64]
        )
        
        # Initialize trainer
        self.trainer = AdvancedTrainer(self.model)
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        shutil.rmtree(self.test_dir)
        
    def test_dataset(self):
        """Test multi-modal dataset."""
        # Create dataset
        dataset = MultiModalDataset(
            self.text_embeddings,
            self.metrics,
            self.feedback
        )
        
        # Check length
        self.assertEqual(len(dataset), self.batch_size)
        
        # Check item
        text, metrics, feedback = dataset[0]
        self.assertEqual(text.shape, (self.text_dim,))
        self.assertEqual(metrics.shape, (self.metric_dim,))
        self.assertEqual(feedback.shape, (self.feedback_dim,))
        
    def test_hierarchical_network(self):
        """Test hierarchical network."""
        # Forward pass
        agent_scores, perf_pred = self.model(
            self.text_embeddings,
            self.metrics,
            self.feedback
        )
        
        # Check output shapes
        self.assertEqual(agent_scores.shape, (self.batch_size, 1))
        self.assertEqual(perf_pred.shape, (self.batch_size, self.metric_dim))
        
    def test_trainer_epoch(self):
        """Test training epoch."""
        # Create dataset and loader
        dataset = MultiModalDataset(
            self.text_embeddings,
            self.metrics,
            self.feedback
        )
        loader = DataLoader(dataset, batch_size=2)
        
        # Train epoch
        metrics = self.trainer.train_epoch(loader, epoch=0)
        
        # Check metrics
        self.assertIn("total_loss", metrics)
        self.assertIn("agent_loss", metrics)
        self.assertIn("perf_loss", metrics)
        
    def test_validation(self):
        """Test validation."""
        # Create dataset and loader
        dataset = MultiModalDataset(
            self.text_embeddings,
            self.metrics,
            self.feedback
        )
        loader = DataLoader(dataset, batch_size=2)
        
        # Run validation
        metrics = self.trainer.validate(loader)
        
        # Check metrics
        self.assertIn("val_total_loss", metrics)
        self.assertIn("val_agent_loss", metrics)
        self.assertIn("val_perf_loss", metrics)
        
    def test_full_training(self):
        """Test full training loop."""
        # Create datasets and loaders
        train_dataset = MultiModalDataset(
            self.text_embeddings,
            self.metrics,
            self.feedback
        )
        val_dataset = MultiModalDataset(
            self.text_embeddings,
            self.metrics,
            self.feedback
        )
        
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        
        # Train model
        history = self.trainer.train(
            train_loader,
            val_loader,
            num_epochs=2,
            checkpoint_dir=self.test_dir
        )
        
        # Check history
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("learning_rate", history)
        self.assertEqual(len(history["train_loss"]), 2)
        
        # Check checkpoint
        checkpoint_path = f"{self.test_dir}/best_model.pt"
        self.assertTrue(torch.load(checkpoint_path))
        
    def test_checkpoint_loading(self):
        """Test checkpoint loading."""
        # Create checkpoint
        checkpoint = {
            "epoch": 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "scheduler_state_dict": self.trainer.scheduler.state_dict(),
            "val_loss": 0.5
        }
        
        checkpoint_path = f"{self.test_dir}/test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        self.trainer.load_checkpoint(checkpoint_path)
        
        # Verify loading
        self.mock_mlflow.log_event.assert_called()
        
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid checkpoint path
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_checkpoint("invalid_path")
            
        # Test training with empty loader
        empty_dataset = MultiModalDataset(
            torch.empty(0, self.text_dim),
            torch.empty(0, self.metric_dim),
            torch.empty(0, self.feedback_dim)
        )
        empty_loader = DataLoader(empty_dataset, batch_size=2)
        
        metrics = self.trainer.validate(empty_loader)
        self.assertEqual(metrics["val_total_loss"], 0)

if __name__ == '__main__':
    unittest.main()