import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from core.utils.imports import torch
pytestmark = pytest.mark.heavy
pytestmark = pytest.mark.skipif(torch is None, reason="Torch not installed")
import os
import tempfile
from langchain.schema import Document
from agents.worker_agent import WorkerAgent
from nn_models.agent_nn_v2 import TaskMetrics



class TestWorkerAgentNNFeatures(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock dependencies
        self.llm_patcher = patch("agents.worker_agent.SpecializedLLM")
        self.db_patcher = patch("agents.worker_agent.WorkerAgentDB")
        self.nn_patcher = patch("agents.worker_agent.AgentNN")

        self.mock_llm = self.llm_patcher.start()
        self.mock_db = self.db_patcher.start()
        self.mock_nn = self.nn_patcher.start()

        # Set up mock LLM behavior
        self.mock_llm_instance = MagicMock()
        self.mock_llm_instance.get_embedding.return_value = [
            0.1
        ] * 768  # Mock embedding
        self.mock_llm_instance.generate_with_confidence.return_value = (
            "Test response",
            0.8,
        )
        self.mock_llm.return_value = self.mock_llm_instance

        # Set up mock DB behavior
        self.mock_db_instance = MagicMock()
        self.mock_db_instance.search.return_value = [
            Document(page_content="Test document", metadata={"source": "test"})
        ]
        self.mock_db.return_value = self.mock_db_instance

        # Set up mock NN behavior
        self.mock_nn_instance = MagicMock()
        self.mock_nn_instance.predict_task_features.return_value = torch.tensor(
            [[0.5] * 64]
        )
        self.mock_nn_instance.get_training_summary.return_value = {
            "avg_loss": 0.1,
            "total_batches": 100,
        }
        self.mock_nn.return_value = self.mock_nn_instance

        # Initialize agent
        self.agent = WorkerAgent("test_agent", use_nn_features=True)

    def tearDown(self):
        """Clean up after tests."""
        self.llm_patcher.stop()
        self.db_patcher.stop()
        self.nn_patcher.stop()

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "test_agent")
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.db)
        self.assertIsNotNone(self.agent.nn)

    def test_get_task_embedding(self):
        """Test task embedding generation."""
        embedding = self.agent.get_task_embedding("Test task")
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, (1, 768))  # Batch size 1, embedding size 768

    def test_format_task_features(self):
        """Test task features formatting."""
        features = torch.tensor([[0.1, 0.2, 0.3]])
        formatted = self.agent.format_task_features(features)
        self.assertIsInstance(formatted, str)
        self.assertIn("Feature 1: 0.100", formatted)
        self.assertIn("Feature 2: 0.200", formatted)
        self.assertIn("Feature 3: 0.300", formatted)

    async def test_execute_task(self):
        """Test task execution with neural network enhancement."""
        response = await self.agent.execute_task("Test task")

        # Check that LLM was called with task features
        self.mock_llm_instance.generate_with_confidence.assert_called_once()
        call_args = self.mock_llm_instance.generate_with_confidence.call_args[1]
        self.assertIn("task_features", call_args)

        # Check response
        self.assertEqual(response, "Test response")

        # Check that metrics were recorded
        self.mock_nn_instance.evaluate_performance.assert_called_once()
        metrics = self.mock_nn_instance.evaluate_performance.call_args[0][0]
        self.assertIsInstance(metrics, TaskMetrics)
        self.assertEqual(metrics.confidence_score, 0.8)

    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        # Set up mock metrics
        self.mock_nn_instance.eval_metrics = [
            {"response_time": 0.1, "confidence": 0.8, "user_feedback": 4.5},
            {"response_time": 0.2, "confidence": 0.9, "user_feedback": 4.0},
        ]

        metrics = self.agent.get_performance_metrics()

        # Check metrics
        self.assertIn("avg_loss", metrics)
        self.assertIn("total_batches", metrics)
        self.assertIn("avg_response_time", metrics)
        self.assertIn("avg_confidence", metrics)
        self.assertIn("avg_user_feedback", metrics)

        # Check values
        self.assertEqual(metrics["avg_response_time"], 0.15)
        self.assertEqual(metrics["avg_confidence"], 0.85)
        self.assertEqual(metrics["avg_user_feedback"], 4.25)

    def test_save_load_nn(self):
        """Test saving and loading neural network state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up paths
            os.environ["MODELS_DIR"] = tmpdir
            model_path = os.path.join(tmpdir, "agent_nn", "test_agent_nn.pt")

            # Save model
            self.agent.save_nn()

            # Check that save was called
            self.mock_nn_instance.save_model.assert_called_once_with(model_path)

            # Create new agent and load model
            new_agent = WorkerAgent("test_agent", use_nn_features=True)

            # Check that load was attempted
            self.mock_nn_instance.load_model.assert_called_once_with(model_path)

    def test_knowledge_base_operations(self):
        """Test knowledge base operations."""
        # Test adding documents
        docs = [Document(page_content="Test content", metadata={"source": "test"})]
        self.agent.ingest_knowledge(docs)
        self.mock_db_instance.ingest_documents.assert_called_once()

        # Test searching
        results = self.agent.search_knowledge_base("test query")
        self.mock_db_instance.search.assert_called_once_with("test query", k=4)
        self.assertEqual(len(results), 1)

        # Test clearing
        self.agent.clear_knowledge()
        self.mock_db_instance.clear_knowledge_base.assert_called_once()

    async def test_shutdown(self):
        """Test agent shutdown."""
        # Add mock communication hub
        self.agent.communication_hub = AsyncMock()

        await self.agent.shutdown()

        # Check that model was saved
        self.mock_nn_instance.save_model.assert_called_once()

        # Check that agent was deregistered
        self.agent.communication_hub.deregister_agent.assert_called_once_with(
            "test_agent"
        )

if __name__ == "__main__":
    unittest.main()
