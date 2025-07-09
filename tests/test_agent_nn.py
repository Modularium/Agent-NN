import unittest
import pytest

torch = pytest.importorskip("torch")
import os
import tempfile
from nn_models.agent_nn_v2 import AgentNN, TaskMetrics

pytestmark = pytest.mark.heavy


class TestAgentNN(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.input_size = 768
        self.hidden_size = 256
        self.output_size = 64
        self.model = AgentNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

        # Create sample data
        self.batch_size = 32
        self.task_embedding = torch.randn(self.batch_size, self.input_size)
        self.target_features = torch.randn(self.batch_size, self.output_size)
        self.target_features = torch.tanh(self.target_features)  # Scale to [-1, 1]

    def test_initialization(self):
        """Test model initialization."""
        # Check model structure
        self.assertIsInstance(self.model, AgentNN)
        self.assertEqual(
            len(list(self.model.parameters())), 6
        )  # 3 layers * (weights + biases)

        # Check output shape
        output = self.model(self.task_embedding)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

        # Check output range (should be between -1 and 1 due to tanh)
        self.assertTrue(torch.all(output >= -1))
        self.assertTrue(torch.all(output <= 1))

    def test_training_step(self):
        """Test training step."""
        # Initial loss
        initial_loss = self.model.train_step(self.task_embedding, self.target_features)
        self.assertIsInstance(initial_loss, float)

        # Train for a few steps
        losses = []
        for _ in range(10):
            loss = self.model.train_step(self.task_embedding, self.target_features)
            losses.append(loss)

        # Loss should decrease
        self.assertLess(losses[-1], initial_loss)

        # Check training history
        self.assertEqual(len(self.model.training_losses), 11)  # Initial + 10 steps

    def test_prediction(self):
        """Test prediction functionality."""
        # Get predictions
        predictions = self.model.predict_task_features(self.task_embedding)

        # Check output shape and range
        self.assertEqual(predictions.shape, (self.batch_size, self.output_size))
        self.assertTrue(torch.all(predictions >= -1))
        self.assertTrue(torch.all(predictions <= 1))

        # Check that predict_task_features doesn't modify gradients
        self.assertFalse(predictions.requires_grad)

    def test_evaluation(self):
        """Test performance evaluation."""
        metrics = TaskMetrics(
            response_time=0.5,
            confidence_score=0.8,
            user_feedback=4.5,
            task_success=True,
        )

        eval_results = self.model.evaluate_performance(metrics)

        # Check metrics
        self.assertEqual(eval_results["response_time"], 0.5)
        self.assertEqual(eval_results["confidence"], 0.8)
        self.assertEqual(eval_results["user_feedback"], 4.5)
        self.assertEqual(eval_results["success_rate"], 1.0)

        # Check metrics history
        self.assertEqual(len(self.model.eval_metrics), 1)

    def test_training_summary(self):
        """Test training summary generation."""
        # Train for a few steps
        for _ in range(10):
            self.model.train_step(self.task_embedding, self.target_features)

        summary = self.model.get_training_summary()

        # Check summary contents
        self.assertIn("avg_loss", summary)
        self.assertIn("min_loss", summary)
        self.assertIn("max_loss", summary)
        self.assertIn("total_batches", summary)
        self.assertEqual(summary["total_batches"], 10)

    def test_save_load(self):
        """Test model saving and loading."""
        # Train the model
        initial_loss = self.model.train_step(self.task_embedding, self.target_features)
        initial_predictions = self.model.predict_task_features(self.task_embedding)

        # Save the model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.model.save_model(tmp.name)

            # Create a new model and load the saved state
            loaded_model = AgentNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
            )
            loaded_model.load_model(tmp.name)

        # Clean up
        os.unlink(tmp.name)

        # Check that loaded model produces same predictions
        loaded_predictions = loaded_model.predict_task_features(self.task_embedding)
        self.assertTrue(torch.allclose(initial_predictions, loaded_predictions))

        # Check that training history was loaded
        self.assertEqual(len(loaded_model.training_losses), 1)
        self.assertEqual(loaded_model.training_losses[0], initial_loss)

if __name__ == "__main__":
    unittest.main()
