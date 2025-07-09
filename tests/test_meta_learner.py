import unittest
from unittest.mock import patch, MagicMock
import pytest

torch = pytest.importorskip("torch")
import os
import tempfile
import shutil
from managers.meta_learner import MetaLearner, AgentScore
from nn_models.agent_nn_v2 import TaskMetrics

pytestmark = pytest.mark.heavy


class TestMetaLearner(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch("utils.logging_util.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()
        self.mock_mlflow.active_run.return_value = None

        # Initialize meta-learner
        self.meta_learner = MetaLearner(
            embedding_size=768, feature_size=64, hidden_size=256
        )

        # Create sample data
        self.task_embedding = torch.randn(1, 768)  # Batch size 1
        self.agent_features = torch.randn(1, 64)  # Batch size 1
        self.success_score = torch.tensor([[0.8]])  # High success score

    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()

    def test_initialization(self):
        """Test meta-learner initialization."""
        self.assertIsInstance(self.meta_learner, MetaLearner)
        self.assertEqual(self.meta_learner.embedding_size, 768)
        self.assertEqual(self.meta_learner.feature_size, 64)

        # Check network structure
        self.assertIsNotNone(self.meta_learner.network)
        self.assertIsNotNone(self.meta_learner.optimizer)
        self.assertIsNotNone(self.meta_learner.criterion)

    def test_forward_pass(self):
        """Test forward pass through the network."""
        output = self.meta_learner(self.task_embedding, self.agent_features)

        # Check output shape and range
        self.assertEqual(output.shape, (1, 1))  # Batch size 1, single score
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_training_step(self):
        """Test training step."""
        # Initial training step
        initial_loss = self.meta_learner.train_step(
            self.task_embedding, self.agent_features, self.success_score
        )

        # Train for several steps
        losses = []
        for _ in range(10):
            loss = self.meta_learner.train_step(
                self.task_embedding, self.agent_features, self.success_score
            )
            losses.append(loss)

        # Loss should decrease
        self.assertLess(losses[-1], initial_loss)

        # Check MLflow logging
        self.mock_mlflow.log_metrics.assert_called()

    def test_score_agents(self):
        """Test agent scoring."""
        agents_features = {
            "agent1": torch.randn(1, 64),
            "agent2": torch.randn(1, 64),
            "agent3": torch.randn(1, 64),
        }

        scores = self.meta_learner.score_agents(self.task_embedding, agents_features)

        # Check scores
        self.assertEqual(len(scores), 3)
        self.assertIsInstance(scores[0], AgentScore)

        # Scores should be sorted
        self.assertTrue(
            all(
                scores[i].combined_score >= scores[i + 1].combined_score
                for i in range(len(scores) - 1)
            )
        )

    def test_update_metrics(self):
        """Test metrics updating."""
        metrics = TaskMetrics(
            response_time=0.5,
            confidence_score=0.8,
            user_feedback=4.5,
            task_success=True,
        )

        # Update metrics for an agent
        self.meta_learner.update_metrics("test_agent", metrics)

        # Check metrics were stored
        self.assertIn("test_agent", self.meta_learner.agent_metrics)
        self.assertEqual(len(self.meta_learner.agent_metrics["test_agent"]), 1)

        # Check MLflow logging
        self.mock_mlflow.log_metrics.assert_called()

    def test_historical_performance(self):
        """Test historical performance calculation."""
        # Add some metrics
        metrics = [
            TaskMetrics(
                response_time=0.5,
                confidence_score=0.8,
                user_feedback=4.5,
                task_success=True,
            ),
            TaskMetrics(
                response_time=0.7,
                confidence_score=0.6,
                user_feedback=3.5,
                task_success=False,
            ),
        ]

        for metric in metrics:
            self.meta_learner.update_metrics("test_agent", metric)

        # Get historical performance
        score = self.meta_learner._get_historical_performance("test_agent")

        # Check score
        self.assertIsNotNone(score)
        self.assertTrue(0 <= score <= 1)

        # Check non-existent agent
        score = self.meta_learner._get_historical_performance("nonexistent")
        self.assertIsNone(score)

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "meta_learner.pt")

            # Add some metrics
            metrics = TaskMetrics(
                response_time=0.5,
                confidence_score=0.8,
                user_feedback=4.5,
                task_success=True,
            )
            self.meta_learner.update_metrics("test_agent", metrics)

            # Train model
            self.meta_learner.train_step(
                self.task_embedding, self.agent_features, self.success_score
            )

            # Save model
            self.meta_learner.save_model(model_path)

            # Create new instance and load model
            new_learner = MetaLearner(
                embedding_size=768, feature_size=64, hidden_size=256
            )
            new_learner.load_model(model_path)

            # Check metrics were loaded
            self.assertIn("test_agent", new_learner.agent_metrics)
            self.assertEqual(
                len(new_learner.agent_metrics["test_agent"]),
                len(self.meta_learner.agent_metrics["test_agent"]),
            )

            # Check training history was loaded
            self.assertEqual(
                len(new_learner.training_history),
                len(self.meta_learner.training_history),
            )

if __name__ == "__main__":
    unittest.main()
