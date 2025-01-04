import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os
from datetime import datetime
from managers.specialized_llm_manager import SpecializedLLMManager
from nn_models.agent_nn import TaskMetrics

class TestSpecializedLLMManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.models_path = os.path.join(self.test_dir, "models")
        self.cache_path = os.path.join(self.test_dir, "cache")
        
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Initialize manager
        self.manager = SpecializedLLMManager(
            models_path=self.models_path,
            cache_path=self.cache_path
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        shutil.rmtree(self.test_dir)
        
    def test_add_model(self):
        """Test model addition."""
        # Add model
        self.manager.add_model(
            "test_model",
            "test_domain",
            "gpt-3.5-turbo",
            "Test model"
        )
        
        # Check model was added
        info = self.manager.get_model_info("test_model")
        self.assertEqual(info["domain"], "test_domain")
        self.assertEqual(info["base_model"], "gpt-3.5-turbo")
        
    def test_get_domain_models(self):
        """Test getting domain models."""
        # Add models
        self.manager.add_model(
            "model1",
            "domain1",
            "gpt-3.5-turbo",
            "Model 1"
        )
        self.manager.add_model(
            "model2",
            "domain1",
            "gpt-4",
            "Model 2"
        )
        self.manager.add_model(
            "model3",
            "domain2",
            "gpt-3.5-turbo",
            "Model 3"
        )
        
        # Get domain models
        models = self.manager.get_domain_models("domain1")
        
        # Check results
        self.assertEqual(len(models), 2)
        self.assertIn("model1", models)
        self.assertIn("model2", models)
        
    def test_update_model_status(self):
        """Test model status updates."""
        # Add model
        self.manager.add_model(
            "test_model",
            "test_domain",
            "gpt-3.5-turbo",
            "Test model"
        )
        
        # Update status
        self.manager.update_model_status(
            "test_model",
            "training",
            {"progress": 0.5}
        )
        
        # Check status
        info = self.manager.get_model_info("test_model")
        self.assertEqual(info["training_status"], "training")
        self.assertEqual(info["status_metadata"]["progress"], 0.5)
        
    def test_update_model_metrics(self):
        """Test model metrics updates."""
        # Add model
        self.manager.add_model(
            "test_model",
            "test_domain",
            "gpt-3.5-turbo",
            "Test model"
        )
        
        # Update metrics
        metrics = TaskMetrics(
            response_time=0.5,
            confidence_score=0.8,
            task_success=True
        )
        self.manager.update_model_metrics("test_model", metrics)
        
        # Check metrics
        info = self.manager.get_model_info("test_model")
        self.assertIn("metrics", info)
        self.assertEqual(info["metrics"]["total_tasks"], 1)
        self.assertEqual(info["metrics"]["success_rate"], 1.0)
        
    def test_get_best_model(self):
        """Test best model selection."""
        # Add models
        self.manager.add_model(
            "model1",
            "test_domain",
            "gpt-3.5-turbo",
            "Model 1"
        )
        self.manager.add_model(
            "model2",
            "test_domain",
            "gpt-4",
            "Model 2"
        )
        
        # Add metrics
        metrics1 = TaskMetrics(
            response_time=0.5,
            confidence_score=0.6,
            task_success=True
        )
        metrics2 = TaskMetrics(
            response_time=0.3,
            confidence_score=0.9,
            task_success=True
        )
        
        self.manager.update_model_metrics("model1", metrics1)
        self.manager.update_model_metrics("model2", metrics2)
        
        # Get best model
        best_model = self.manager.get_best_model(
            "test_domain",
            "Test task"
        )
        
        # Check result (model2 should be better)
        self.assertEqual(best_model, "model2")
        
    def test_remove_model(self):
        """Test model removal."""
        # Add model
        self.manager.add_model(
            "test_model",
            "test_domain",
            "gpt-3.5-turbo",
            "Test model"
        )
        
        # Remove model
        self.manager.remove_model("test_model")
        
        # Check model was removed
        with self.assertRaises(ValueError):
            self.manager.get_model_info("test_model")
            
    def test_get_model_stats(self):
        """Test model statistics."""
        # Add models
        self.manager.add_model(
            "model1",
            "domain1",
            "gpt-3.5-turbo",
            "Model 1"
        )
        self.manager.add_model(
            "model2",
            "domain2",
            "gpt-4",
            "Model 2"
        )
        
        # Add metrics
        metrics = TaskMetrics(
            response_time=0.5,
            confidence_score=0.8,
            task_success=True
        )
        self.manager.update_model_metrics("model1", metrics)
        
        # Get stats
        stats = self.manager.get_model_stats()
        
        # Check stats
        self.assertEqual(len(stats), 2)
        self.assertIn("model1", stats)
        self.assertIn("model2", stats)
        self.assertIn("metrics", stats["model1"])

if __name__ == '__main__':
    unittest.main()