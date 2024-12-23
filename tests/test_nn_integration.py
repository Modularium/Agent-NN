"""Tests for the neural network integration and agent selection."""
import os
import unittest
import numpy as np
from datetime import datetime
from managers.nn_manager import NNManager
from utils.agent_descriptions import (
    get_agent_description,
    get_task_requirements,
    match_task_to_domain
)
from mlflow_integration.experiment_tracking import ExperimentTracker

class TestNNIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set")
            
    def setUp(self):
        self.nn_manager = NNManager()
        self.available_agents = [
            "finance_agent",
            "tech_agent",
            "marketing_agent"
        ]
        
    def test_agent_selection(self):
        """Test agent selection for different types of tasks."""
        # Test financial task
        financial_task = "Calculate the ROI for our new investment portfolio"
        agent = self.nn_manager.predict_best_agent(financial_task, self.available_agents)
        self.assertEqual(agent, "finance_agent")
        
        # Test technical task
        tech_task = "Implement a new microservice using Python and Docker"
        agent = self.nn_manager.predict_best_agent(tech_task, self.available_agents)
        self.assertEqual(agent, "tech_agent")
        
        # Test marketing task
        marketing_task = "Create a social media campaign for our new product"
        agent = self.nn_manager.predict_best_agent(marketing_task, self.available_agents)
        self.assertEqual(agent, "marketing_agent")
        
    def test_embedding_cache(self):
        """Test that embeddings are properly cached."""
        test_text = "This is a test text"
        
        # Get embedding first time
        embedding1 = self.nn_manager._get_embedding(test_text)
        
        # Get embedding second time (should use cache)
        embedding2 = self.nn_manager._get_embedding(test_text)
        
        # Verify embeddings are identical
        np.testing.assert_array_equal(embedding1, embedding2)
        
        # Verify text is in cache
        self.assertIn(test_text, self.nn_manager.embedding_cache)
        
    def test_task_requirements(self):
        """Test task requirement extraction."""
        task = "Analyze the market trends and create a report"
        requirements = get_task_requirements(task)
        
        self.assertIn("analytical_capability", requirements)
        self.assertIn("creative_capability", requirements)
        
    def test_domain_matching(self):
        """Test domain matching functionality."""
        requirements = ["analytical_capability", "creative_capability"]
        
        # Test finance domain match
        finance_score = match_task_to_domain(requirements, "finance")
        self.assertGreater(finance_score, 0)
        
        # Test marketing domain match
        marketing_score = match_task_to_domain(requirements, "marketing")
        self.assertGreater(marketing_score, 0)
        
    def test_model_updates(self):
        """Test model parameter updates based on performance."""
        # Initial parameters
        initial_threshold = self.nn_manager.confidence_threshold
        initial_weights = self.nn_manager.weights.copy()
        
        # Simulate successful task execution
        self.nn_manager.update_model(
            task_description="Test task",
            chosen_agent="finance_agent",
            execution_result={
                "success": True,
                "execution_time": 10,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Verify parameters were updated
        self.assertNotEqual(initial_threshold, self.nn_manager.confidence_threshold)
        
    def test_performance_tracking(self):
        """Test performance tracking and metrics calculation."""
        # Execute some test tasks
        for i in range(5):
            self.nn_manager.update_model(
                task_description=f"Test task {i}",
                chosen_agent="tech_agent",
                execution_result={
                    "success": i % 2 == 0,  # Alternate success/failure
                    "execution_time": 10 + i,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        # Get performance metrics
        metrics = self.nn_manager.get_performance_metrics()
        
        # Verify metrics structure
        self.assertIn("model_parameters", metrics)
        self.assertIn("performance_metrics", metrics)
        self.assertEqual(metrics["total_tasks_processed"], 5)
        
    def test_experiment_tracking(self):
        """Test MLflow experiment tracking integration."""
        tracker = ExperimentTracker("test_experiment")
        
        # Log a test run
        with tracker.start_run("test_run") as run:
            tracker.log_agent_selection(
                task_description="Test task",
                chosen_agent="finance_agent",
                available_agents=self.available_agents,
                agent_scores={"finance_agent": 0.8, "tech_agent": 0.5},
                execution_result={"success": True, "execution_time": 15}
            )
            
        # Get best run
        best_run = tracker.get_best_run("success_score")
        self.assertIsNotNone(best_run)
        
    def test_error_handling(self):
        """Test error handling in agent selection."""
        # Test with empty agent list
        result = self.nn_manager.predict_best_agent("Test task", [])
        self.assertIsNone(result)
        
        # Test with invalid task description
        result = self.nn_manager.predict_best_agent("", self.available_agents)
        self.assertIsNone(result)
        
    def test_agent_descriptions(self):
        """Test agent description functionality."""
        # Test getting description for known domain
        finance_desc = get_agent_description("finance")
        self.assertIn("capabilities", finance_desc)
        self.assertIn("knowledge_domains", finance_desc)
        
        # Test getting description for unknown domain
        unknown_desc = get_agent_description("unknown_domain")
        self.assertIn("capabilities", unknown_desc)
        self.assertEqual(len(unknown_desc["capabilities"]), 1)
        
if __name__ == '__main__':
    unittest.main()