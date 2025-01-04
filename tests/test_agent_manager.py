import unittest
from unittest.mock import patch, MagicMock
from managers.agent_manager import AgentManager, WorkerAgent
from langchain.schema import Document

class TestAgentManager(unittest.TestCase):
    def setUp(self):
        # Mock OpenAI embeddings to avoid API calls during tests
        self.embeddings_patcher = patch('managers.agent_manager.OpenAIEmbeddings')
        self.mock_embeddings = self.embeddings_patcher.start()
        
        # Create a mock embeddings instance
        self.mock_embeddings_instance = MagicMock()
        self.mock_embeddings.return_value = self.mock_embeddings_instance
        
        # Set up mock embeddings behavior
        def mock_embed_query(text):
            # Return different mock embeddings for different domains
            if "finance" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "tech" in text.lower():
                return [0.0, 1.0, 0.0]
            elif "marketing" in text.lower():
                return [0.0, 0.0, 1.0]
            return [0.3, 0.3, 0.3]  # Default embedding
        
        self.mock_embeddings_instance.embed_query.side_effect = mock_embed_query
        
        # Mock WorkerAgent to avoid LLM initialization
        self.worker_agent_patcher = patch('managers.agent_manager.WorkerAgent')
        self.mock_worker_agent = self.worker_agent_patcher.start()
        
        # Create mock worker agent instances
        def create_mock_agent(name, domain_docs=None):
            mock_agent = MagicMock()
            mock_agent.name = name
            mock_agent.search_knowledge_base.return_value = domain_docs or []
            return mock_agent
        
        self.mock_worker_agent.side_effect = create_mock_agent
        
        # Initialize AgentManager
        self.agent_manager = AgentManager()

    def tearDown(self):
        self.embeddings_patcher.stop()
        self.worker_agent_patcher.stop()

    def test_initialization(self):
        """Test that AgentManager initializes with default agents"""
        agents = self.agent_manager.get_all_agents()
        
        # Check that default agents are created
        self.assertEqual(len(agents), 3)
        self.assertIn("finance_agent", agents)
        self.assertIn("tech_agent", agents)
        self.assertIn("marketing_agent", agents)

    def test_get_agent(self):
        """Test getting a specific agent"""
        # Get existing agent
        agent = self.agent_manager.get_agent("finance_agent")
        self.assertIsInstance(agent, WorkerAgent)
        self.assertEqual(agent.name, "finance")
        
        # Get non-existent agent
        agent = self.agent_manager.get_agent("nonexistent_agent")
        self.assertIsNone(agent)

    def test_create_new_agent(self):
        """Test creating a new agent based on task description"""
        # Create new finance agent
        task_description = "Create a financial analysis report for Q4 2023"
        new_agent = self.agent_manager.create_new_agent(task_description)
        
        # Verify agent creation
        self.assertIsInstance(new_agent, WorkerAgent)
        self.assertEqual(new_agent.name, "finance")
        
        # Verify agent is added to manager
        agents = self.agent_manager.get_all_agents()
        self.assertEqual(len(agents), 4)  # 3 default + 1 new
        self.assertTrue(any("finance_agent_" in name for name in agents))

    def test_infer_domain(self):
        """Test domain inference from task description"""
        # Test finance domain
        domain = self.agent_manager._infer_domain("Create a budget analysis")
        self.assertEqual(domain, "finance")
        
        # Test tech domain
        domain = self.agent_manager._infer_domain("Debug the Python application")
        self.assertEqual(domain, "tech")
        
        # Test marketing domain
        domain = self.agent_manager._infer_domain("Create a social media campaign")
        self.assertEqual(domain, "marketing")

    def test_get_agent_metadata(self):
        """Test getting agent metadata"""
        # Get metadata for existing agent
        metadata = self.agent_manager.get_agent_metadata("finance_agent")
        self.assertEqual(metadata["name"], "finance_agent")
        self.assertEqual(metadata["domain"], "finance")
        self.assertGreater(metadata["knowledge_base_size"], 0)
        
        # Get metadata for non-existent agent
        metadata = self.agent_manager.get_agent_metadata("nonexistent_agent")
        self.assertEqual(metadata, {})

if __name__ == '__main__':
    unittest.main()