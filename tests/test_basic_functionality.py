import os
import unittest
from agents.supervisor_agent import SupervisorAgent
from agents.chatbot_agent import ChatbotAgent
from agents.worker_agent import WorkerAgent
from managers.agent_manager import AgentManager
from managers.nn_manager import NNManager
from datastores.vector_store import VectorStore
from datastores.worker_agent_db import WorkerAgentDB
from utils.logging_util import setup_logger

class TestBasicFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set")
            
        # Set up test logger
        cls.logger = setup_logger("test_logger", "logs/test.log")
        
    def setUp(self):
        self.supervisor = SupervisorAgent()
        self.chatbot = ChatbotAgent(self.supervisor)
        
    def test_agent_creation(self):
        """Test that default agents are created correctly."""
        agent_manager = AgentManager()
        agents = agent_manager.get_all_agents()
        
        # Check that default agents exist
        self.assertIn("finance_agent", agents)
        self.assertIn("tech_agent", agents)
        self.assertIn("marketing_agent", agents)
        
    def test_vector_store(self):
        """Test vector store functionality."""
        store = VectorStore("test_collection")
        
        # Test adding and retrieving documents
        test_docs = [
            "This is a test document about finance.",
            "This is a test document about technology."
        ]
        store.add_documents([{"page_content": doc, "metadata": {}} for doc in test_docs])
        
        # Test similarity search
        results = store.similarity_search("finance", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("finance", results[0].page_content.lower())
        
    def test_worker_agent(self):
        """Test worker agent functionality."""
        agent = WorkerAgent("test_agent", [
            "Test agent specializes in handling test-related tasks.",
            "It can process and validate test data."
        ])
        
        # Test task execution
        result = agent.execute_task("What can you help me with?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
    def test_chatbot_task_identification(self):
        """Test chatbot's ability to identify tasks."""
        # Test with a clear task
        response = self.chatbot.handle_user_message(
            "Please analyze the financial report for Q1 2024."
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Test with casual conversation
        response = self.chatbot.handle_user_message(
            "How are you doing today?"
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
    def test_supervisor_task_execution(self):
        """Test supervisor's task execution and agent selection."""
        # Test financial task
        result = self.supervisor.execute_task(
            "Calculate the ROI for a project with $1000 investment and $1500 return."
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        
        # Test technical task
        result = self.supervisor.execute_task(
            "Explain how to implement a binary search algorithm."
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        
    def test_error_handling(self):
        """Test error handling in various components."""
        # Test invalid API key scenario20241022
        original_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "invalid_key"
        
        try:
            response = self.chatbot.handle_user_message("Hello")
            self.assertIn("error", response.lower())
        finally:
            # Restore original key
            os.environ["OPENAI_API_KEY"] = original_key
            
    def test_conversation_history(self):
        """Test chatbot's conversation history management."""
        messages = [
            "Hello!",
            "How can you help me?",
            "What services do you offer?"
        ]
        
        for msg in messages:
            self.chatbot.handle_user_message(msg)
            
        summary = self.chatbot.get_conversation_summary()
        self.assertEqual(summary["user_messages"], len(messages))
        self.assertEqual(summary["assistant_messages"], len(messages))
        
    def tearDown(self):
        # Clean up any test data
        self.chatbot.clear_history()

if __name__ == '__main__':
    unittest.main()