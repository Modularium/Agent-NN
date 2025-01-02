import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import tempfile
import os
import json
from datetime import datetime, timedelta
from managers.ab_testing import (
    TestStatus,
    SignificanceLevel,
    Variant,
    TestResult,
    ABTest,
    ABTestingManager
)

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

class TestABTesting(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('managers.ab_testing.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment
        
        # Create test models
        self.model_a = SimpleModel()
        self.model_b = SimpleModel()
        
        # Create test variants
        self.variants = [
            Variant(
                name="variant_a",
                model=self.model_a,
                config={"learning_rate": 0.001},
                traffic_split=0.5
            ),
            Variant(
                name="variant_b",
                model=self.model_b,
                config={"learning_rate": 0.002},
                traffic_split=0.5
            )
        ]
        
        # Create test metrics
        self.metrics = ["accuracy", "loss"]
        
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize manager
        self.manager = ABTestingManager()
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        import shutil
        shutil.rmtree(self.test_dir)
        
    def test_test_creation(self):
        """Test A/B test creation."""
        # Create test
        test = self.manager.create_test(
            "test_1",
            self.variants,
            self.metrics
        )
        
        # Check test
        self.assertEqual(test.test_id, "test_1")
        self.assertEqual(test.status, TestStatus.PENDING)
        self.assertEqual(len(test.variants), 2)
        
        # Test invalid traffic split
        invalid_variants = [
            Variant("a", None, {}, 0.6),
            Variant("b", None, {}, 0.6)
        ]
        with self.assertRaises(ValueError):
            self.manager.create_test(
                "test_2",
                invalid_variants,
                self.metrics
            )
            
    def test_test_lifecycle(self):
        """Test A/B test lifecycle."""
        # Create and start test
        test_id = "test_1"
        test = self.manager.create_test(
            test_id,
            self.variants,
            self.metrics
        )
        self.manager.start_test(test_id)
        
        # Check status
        self.assertEqual(test.status, TestStatus.RUNNING)
        self.assertIsNotNone(test.start_time)
        
        # Add results
        for _ in range(5):
            self.manager.add_result(
                test_id,
                "variant_a",
                {"accuracy": 0.8, "loss": 0.2}
            )
            self.manager.add_result(
                test_id,
                "variant_b",
                {"accuracy": 0.9, "loss": 0.1}
            )
            
        # Stop test
        self.manager.stop_test(test_id)
        self.assertEqual(test.status, TestStatus.STOPPED)
        self.assertIsNotNone(test.end_time)
        
    def test_result_analysis(self):
        """Test result analysis."""
        # Create and start test
        test_id = "test_1"
        self.manager.create_test(
            test_id,
            self.variants,
            self.metrics,
            min_samples=5
        )
        self.manager.start_test(test_id)
        
        # Add results with clear difference
        for _ in range(10):
            self.manager.add_result(
                test_id,
                "variant_a",
                {"accuracy": 0.7, "loss": 0.3}
            )
            self.manager.add_result(
                test_id,
                "variant_b",
                {"accuracy": 0.9, "loss": 0.1}
            )
            
        # Get results
        results = self.manager.get_test_results(test_id)
        
        # Check metrics
        self.assertIn("variants", results)
        self.assertIn("analysis", results)
        
        # Check statistical analysis
        analysis = results["analysis"]
        self.assertIn("accuracy", analysis)
        self.assertIn("loss", analysis)
        self.assertTrue(analysis["accuracy"]["significant"])
        self.assertEqual(analysis["accuracy"]["winner"], "variant_b")
        
    def test_state_persistence(self):
        """Test state saving and loading."""
        # Create and start test
        test_id = "test_1"
        self.manager.create_test(
            test_id,
            self.variants,
            self.metrics
        )
        self.manager.start_test(test_id)
        
        # Add some results
        self.manager.add_result(
            test_id,
            "variant_a",
            {"accuracy": 0.8, "loss": 0.2}
        )
        
        # Save state
        state_path = os.path.join(self.test_dir, "state.json")
        self.manager.save_state(state_path)
        
        # Create new manager and load state
        new_manager = ABTestingManager()
        new_manager.load_state(state_path)
        
        # Check loaded state
        self.assertIn(test_id, new_manager.active_tests)
        loaded_test = new_manager.active_tests[test_id]
        self.assertEqual(loaded_test.status, TestStatus.RUNNING)
        self.assertEqual(
            len(loaded_test.results["variant_a"]),
            1
        )
        
    def test_test_completion(self):
        """Test automatic test completion."""
        # Create test with low min_samples
        test_id = "test_1"
        self.manager.create_test(
            test_id,
            self.variants,
            self.metrics,
            min_samples=2
        )
        self.manager.start_test(test_id)
        
        # Add results until completion
        for _ in range(3):
            self.manager.add_result(
                test_id,
                "variant_a",
                {"accuracy": 0.8, "loss": 0.2}
            )
            
        # Check completion
        test = self.manager.active_tests[test_id]
        self.assertEqual(test.status, TestStatus.COMPLETED)
        
    def test_error_handling(self):
        """Test error handling."""
        test_id = "test_1"
        
        # Test unknown test
        with self.assertRaises(ValueError):
            self.manager.start_test("unknown")
            
        # Test double start
        self.manager.create_test(
            test_id,
            self.variants,
            self.metrics
        )
        self.manager.start_test(test_id)
        with self.assertRaises(ValueError):
            self.manager.start_test(test_id)
            
        # Test unknown variant
        with self.assertRaises(ValueError):
            self.manager.add_result(
                test_id,
                "unknown",
                {"accuracy": 0.8}
            )
            
        # Test adding result to stopped test
        self.manager.stop_test(test_id)
        with self.assertRaises(ValueError):
            self.manager.add_result(
                test_id,
                "variant_a",
                {"accuracy": 0.8}
            )
            
    def test_duration_based_completion(self):
        """Test duration-based test completion."""
        # Create test with short duration
        test_id = "test_1"
        test = self.manager.create_test(
            test_id,
            self.variants,
            self.metrics,
            max_duration=1  # 1 day
        )
        self.manager.start_test(test_id)
        
        # Simulate passage of time
        test.start_time = datetime.now() - timedelta(days=2)
        
        # Add result to trigger completion check
        self.manager.add_result(
            test_id,
            "variant_a",
            {"accuracy": 0.8}
        )
        
        # Check completion
        self.assertEqual(test.status, TestStatus.COMPLETED)

if __name__ == '__main__':
    unittest.main()