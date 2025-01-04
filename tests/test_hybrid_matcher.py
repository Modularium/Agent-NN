import unittest
from unittest.mock import patch, MagicMock
import torch
import os
import tempfile
import shutil
from managers.hybrid_matcher import HybridMatcher, MatchResult
from nn_models.agent_nn import TaskMetrics

class TestHybridMatcher(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        self.mock_mlflow.active_run.return_value = None
        
        # Initialize matcher
        self.matcher = HybridMatcher(
            embedding_size=768,
            feature_size=64,
            similarity_threshold=0.7
        )
        
        # Create sample data
        self.task_description = "Analyze financial data"
        self.task_embedding = torch.randn(1, 768)
        
        self.available_agents = {
            "finance_agent": {
                "embedding": torch.randn(1, 768),
                "features": torch.randn(1, 64)
            },
            "tech_agent": {
                "embedding": torch.randn(1, 768),
                "features": torch.randn(1, 64)
            },
            "marketing_agent": {
                "embedding": torch.randn(1, 768),
                "features": torch.randn(1, 64)
            }
        }

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()

    def test_initialization(self):
        """Test matcher initialization."""
        self.assertIsInstance(self.matcher, HybridMatcher)
        self.assertEqual(self.matcher.embedding_size, 768)
        self.assertEqual(self.matcher.feature_size, 64)
        self.assertEqual(self.matcher.similarity_threshold, 0.7)

    def test_match_task(self):
        """Test task matching."""
        results = self.matcher.match_task(
            self.task_description,
            self.task_embedding,
            self.available_agents
        )
        
        # Check results
        self.assertEqual(len(results), 3)  # One for each agent
        self.assertIsInstance(results[0], MatchResult)
        
        # Check result properties
        for result in results:
            self.assertIn(result.agent_name, self.available_agents)
            self.assertTrue(0 <= result.similarity_score <= 1)
            self.assertTrue(0 <= result.nn_score <= 1)
            self.assertTrue(0 <= result.combined_score <= 1)
            self.assertTrue(0 <= result.confidence <= 1)
            
        # Results should be sorted by combined score
        self.assertTrue(
            all(results[i].combined_score >= results[i+1].combined_score
                for i in range(len(results)-1))
        )

    def test_update_performance(self):
        """Test performance updating."""
        metrics = TaskMetrics(
            response_time=0.5,
            confidence_score=0.8,
            user_feedback=4.5,
            task_success=True
        )
        
        # Update performance
        self.matcher.update_agent_performance(
            "finance_agent",
            metrics,
            success_score=0.9
        )
        
        # Check MLflow logging
        self.mock_mlflow.log_metrics.assert_called()

    def test_meta_learner_training(self):
        """Test meta-learner training."""
        # Create training data
        task_embeddings = torch.randn(10, 768)  # 10 samples
        agent_features = torch.randn(10, 64)
        success_scores = torch.rand(10, 1)  # Random scores between 0 and 1
        
        # Train meta-learner
        self.matcher.train_meta_learner(
            task_embeddings,
            agent_features,
            success_scores,
            num_epochs=5
        )
        
        # Check MLflow logging
        self.assertTrue(
            any("loss" in call[0][0]
                for call in self.mock_mlflow.log_metrics.call_args_list)
        )

    def test_score_combination(self):
        """Test score combination logic."""
        similarity = 0.8
        nn_score = 0.7
        
        # Test without historical data
        combined = self.matcher._combine_scores(
            similarity,
            nn_score,
            "new_agent"  # Agent with no history
        )
        
        self.assertTrue(0 <= combined <= 1)
        
        # Add some history and test again
        metrics = TaskMetrics(
            response_time=0.5,
            confidence_score=0.8,
            user_feedback=4.5,
            task_success=True
        )
        self.matcher.update_agent_performance("test_agent", metrics, 0.9)
        
        combined = self.matcher._combine_scores(
            similarity,
            nn_score,
            "test_agent"  # Agent with history
        )
        
        self.assertTrue(0 <= combined <= 1)

    def test_confidence_calculation(self):
        """Test confidence calculation."""
        similarity = 0.8
        nn_score = 0.7
        combined_score = 0.75
        
        confidence = self.matcher._calculate_confidence(
            similarity,
            nn_score,
            combined_score
        )
        
        # Check confidence score
        self.assertTrue(0 <= confidence <= 1)
        
        # Test with inconsistent scores
        confidence_inconsistent = self.matcher._calculate_confidence(
            0.9,
            0.2,
            0.5
        )
        
        # Confidence should be lower with inconsistent scores
        self.assertLess(confidence_inconsistent, confidence)

    def test_save_load_state(self):
        """Test state saving and loading."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "matcher_state.pt")
            
            # Add some data
            metrics = TaskMetrics(
                response_time=0.5,
                confidence_score=0.8,
                user_feedback=4.5,
                task_success=True
            )
            self.matcher.update_agent_performance("test_agent", metrics, 0.9)
            
            # Save state
            self.matcher.save_state(state_path)
            
            # Create new matcher and load state
            new_matcher = HybridMatcher(
                embedding_size=768,
                feature_size=64
            )
            new_matcher.load_state(state_path)
            
            # Check that historical data was loaded
            self.assertEqual(
                self.matcher._get_historical_performance("test_agent"),
                new_matcher._get_historical_performance("test_agent")
            )

if __name__ == '__main__':
    unittest.main()