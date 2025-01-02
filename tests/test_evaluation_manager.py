import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json
from datetime import datetime, timedelta
from managers.evaluation_manager import (
    EvaluationManager,
    EvaluationMetrics
)

class TestEvaluationManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('managers.evaluation_manager.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment
        
        # Initialize manager
        self.manager = EvaluationManager(
            experiment_name="test_evaluation"
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        
    def test_agent_metrics(self):
        """Test agent metrics tracking."""
        # Create metrics
        metrics = EvaluationMetrics(
            response_time=0.5,
            token_count=100,
            api_cost=0.02,
            success_rate=0.95,
            user_rating=4.5
        )
        
        # Add metrics
        self.manager.add_agent_metrics("agent1", metrics)
        
        # Get performance
        stats = self.manager.get_agent_performance("agent1")
        
        # Check statistics
        self.assertEqual(stats["response_time"]["mean"], 0.5)
        self.assertEqual(stats["success_rate"], 0.95)
        self.assertEqual(stats["total_cost"], 0.02)
        self.assertEqual(stats["user_rating"]["mean"], 4.5)
        
    def test_system_metrics(self):
        """Test system metrics tracking."""
        # Add metrics
        self.manager.add_system_metrics({
            "cpu_usage": 50.0,
            "memory_usage": 1024,
            "active_agents": 5
        })
        
        # Get performance
        stats = self.manager.get_system_performance()
        
        # Check statistics
        self.assertEqual(stats["cpu_usage"]["mean"], 50.0)
        self.assertEqual(stats["memory_usage"]["mean"], 1024)
        self.assertEqual(stats["active_agents"]["mean"], 5)
        
    def test_ab_testing(self):
        """Test A/B testing."""
        # Start test
        test = self.manager.start_ab_test(
            "test_1",
            ["variant_a", "variant_b"],
            ["response_time", "success_rate"],
            duration=24
        )
        
        # Add results
        self.manager.add_ab_test_result(
            "test_1",
            "variant_a",
            {
                "response_time": 0.5,
                "success_rate": 0.9
            }
        )
        self.manager.add_ab_test_result(
            "test_1",
            "variant_b",
            {
                "response_time": 0.6,
                "success_rate": 0.85
            }
        )
        
        # Get results
        results = self.manager.get_ab_test_results("test_1")
        
        # Check results
        self.assertEqual(
            results["statistics"]["variant_a"]["response_time"]["mean"],
            0.5
        )
        self.assertEqual(
            results["statistics"]["variant_b"]["success_rate"]["mean"],
            0.85
        )
        
    def test_cost_tracking(self):
        """Test cost tracking."""
        # Track costs
        self.manager.track_cost(
            0.10,
            "openai",
            {"model": "gpt-4"}
        )
        self.manager.track_cost(
            0.05,
            "openai",
            {"model": "gpt-3.5-turbo"}
        )
        
        # Get analysis
        analysis = self.manager.get_cost_analysis()
        
        # Check analysis
        self.assertEqual(analysis["total_cost"], 0.15)
        self.assertEqual(analysis["by_service"]["openai"], 0.15)
        
    def test_metrics_export_import(self):
        """Test metrics export and import."""
        # Add some metrics
        metrics = EvaluationMetrics(
            response_time=0.5,
            token_count=100,
            api_cost=0.02,
            success_rate=0.95
        )
        self.manager.add_agent_metrics("agent1", metrics)
        
        # Export metrics
        with tempfile.NamedTemporaryFile(mode="w") as f:
            self.manager.export_metrics(f.name)
            
            # Create new manager
            new_manager = EvaluationManager()
            
            # Import metrics
            new_manager.import_metrics(f.name)
            
            # Check imported metrics
            stats = new_manager.get_agent_performance("agent1")
            self.assertEqual(stats["response_time"]["mean"], 0.5)
            
    def test_time_window_filtering(self):
        """Test time window filtering."""
        # Add metrics at different times
        old_metrics = EvaluationMetrics(
            response_time=0.5,
            token_count=100,
            api_cost=0.02,
            success_rate=0.95
        )
        old_metrics.timestamp = datetime.now() - timedelta(hours=25)
        
        new_metrics = EvaluationMetrics(
            response_time=0.6,
            token_count=120,
            api_cost=0.03,
            success_rate=0.90
        )
        
        # Add metrics
        self.manager.add_agent_metrics("agent1", old_metrics)
        self.manager.add_agent_metrics("agent1", new_metrics)
        
        # Get performance with 24h window
        stats = self.manager.get_agent_performance("agent1", window=24)
        
        # Should only include new metrics
        self.assertEqual(stats["response_time"]["mean"], 0.6)
        
    def test_error_handling(self):
        """Test error handling."""
        # Test unknown agent
        stats = self.manager.get_agent_performance("unknown")
        self.assertEqual(stats, {})
        
        # Test unknown test
        with self.assertRaises(ValueError):
            self.manager.get_ab_test_results("unknown")
            
        # Test invalid variant
        with self.assertRaises(ValueError):
            self.manager.add_ab_test_result(
                "test_1",
                "unknown",
                {"metric": 1.0}
            )

if __name__ == '__main__':
    unittest.main()