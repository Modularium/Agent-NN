import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os
import json
from datetime import datetime
from managers.adaptive_learning_manager import AdaptiveLearningManager

class TestAdaptiveLearningManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Mock MLflow
        self.mlflow_patcher = patch('managers.adaptive_learning_manager.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment
        
        # Initialize manager
        self.manager = AdaptiveLearningManager(
            models_path=self.test_dir
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        shutil.rmtree(self.test_dir)
        
    def test_create_experiment(self):
        """Test experiment creation."""
        # Create experiment
        self.manager.create_experiment(
            "test_exp",
            "test_model",
            "Test experiment",
            {"param1": 1, "param2": 2}
        )
        
        # Check experiment was created
        self.assertIn("test_exp", self.manager.experiments)
        exp = self.manager.experiments["test_exp"]
        self.assertEqual(exp["model_id"], "test_model")
        self.assertEqual(exp["parameters"]["param1"], 1)
        
    def test_create_model_variant(self):
        """Test model variant creation."""
        # Create experiment
        self.manager.create_experiment(
            "test_exp",
            "test_model",
            "Test experiment",
            {"param1": 1}
        )
        
        # Create variant
        self.manager.create_model_variant(
            "test_exp",
            "variant1",
            {"hidden_size": 256}
        )
        
        # Check variant was created
        self.assertIn("variant1", self.manager.model_variants)
        variant = self.manager.model_variants["variant1"]
        self.assertEqual(variant["experiment_id"], "test_exp")
        self.assertEqual(variant["architecture"]["hidden_size"], 256)
        
    def test_update_variant_metrics(self):
        """Test variant metrics updates."""
        # Create experiment and variant
        self.manager.create_experiment(
            "test_exp",
            "test_model",
            "Test experiment",
            {"param1": 1}
        )
        self.manager.create_model_variant(
            "test_exp",
            "variant1",
            {"hidden_size": 256}
        )
        
        # Update metrics
        metrics = {"loss": 0.5, "accuracy": 0.9}
        self.manager.update_variant_metrics("variant1", metrics)
        
        # Check metrics were updated
        variant = self.manager.model_variants["variant1"]
        self.assertEqual(variant["metrics"]["loss"], 0.5)
        self.assertEqual(variant["metrics"]["accuracy"], 0.9)
        
        # Check history
        history = self.manager.performance_history["variant1"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["metrics"], metrics)
        
    def test_get_best_variant(self):
        """Test best variant selection."""
        # Create experiment and variants
        self.manager.create_experiment(
            "test_exp",
            "test_model",
            "Test experiment",
            {"param1": 1}
        )
        self.manager.create_model_variant(
            "test_exp",
            "variant1",
            {"hidden_size": 256}
        )
        self.manager.create_model_variant(
            "test_exp",
            "variant2",
            {"hidden_size": 512}
        )
        
        # Add metrics
        self.manager.update_variant_metrics(
            "variant1",
            {"accuracy": 0.8}
        )
        self.manager.update_variant_metrics(
            "variant2",
            {"accuracy": 0.9}
        )
        
        # Get best variant
        best = self.manager.get_best_variant("test_exp", "accuracy")
        
        # Check result
        self.assertEqual(best, "variant2")
        
    def test_optimize_architecture(self):
        """Test architecture optimization."""
        # Create experiment
        self.manager.create_experiment(
            "test_exp",
            "test_model",
            "Test experiment",
            {
                "hidden_sizes": [256, 128],
                "dropout": 0.2,
                "learning_rate": 0.001
            }
        )
        
        # Run optimization
        best_variant, architecture = self.manager.optimize_architecture(
            "test_exp",
            "accuracy",
            num_trials=3
        )
        
        # Check results
        self.assertIsNotNone(best_variant)
        self.assertIsNotNone(architecture)
        self.assertIn("hidden_sizes", architecture)
        self.assertIn("dropout", architecture)
        self.assertIn("learning_rate", architecture)
        
    def test_get_experiment_progress(self):
        """Test experiment progress tracking."""
        # Create experiment and variants
        self.manager.create_experiment(
            "test_exp",
            "test_model",
            "Test experiment",
            {"param1": 1}
        )
        self.manager.create_model_variant(
            "test_exp",
            "variant1",
            {"hidden_size": 256}
        )
        self.manager.create_model_variant(
            "test_exp",
            "variant2",
            {"hidden_size": 512}
        )
        
        # Add metrics
        self.manager.update_variant_metrics(
            "variant1",
            {"accuracy": 0.8, "loss": 0.3}
        )
        self.manager.update_variant_metrics(
            "variant2",
            {"accuracy": 0.9, "loss": 0.2}
        )
        
        # Get progress
        progress = self.manager.get_experiment_progress("test_exp")
        
        # Check statistics
        self.assertEqual(progress["total_variants"], 2)
        self.assertIn("accuracy", progress["metrics"])
        self.assertIn("loss", progress["metrics"])
        
    def test_save_load_experiment(self):
        """Test experiment saving and loading."""
        # Create experiment and variants
        self.manager.create_experiment(
            "test_exp",
            "test_model",
            "Test experiment",
            {"param1": 1}
        )
        self.manager.create_model_variant(
            "test_exp",
            "variant1",
            {"hidden_size": 256}
        )
        
        # Add metrics
        self.manager.update_variant_metrics(
            "variant1",
            {"accuracy": 0.8}
        )
        
        # Save experiment
        self.manager.save_experiment("test_exp")
        
        # Create new manager and load experiment
        new_manager = AdaptiveLearningManager(
            models_path=self.test_dir
        )
        new_manager.load_experiment("test_exp")
        
        # Check loaded data
        self.assertIn("test_exp", new_manager.experiments)
        self.assertIn("variant1", new_manager.model_variants)
        self.assertEqual(
            new_manager.model_variants["variant1"]["metrics"]["accuracy"],
            0.8
        )

if __name__ == '__main__':
    unittest.main()