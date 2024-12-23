"""Tests for agent generation system."""
import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock
import json
from datetime import datetime
from pathlib import Path

from agents.agent_factory import AgentFactory, AgentSpecification
from agents.agent_generator import AgentGenerator
from agents.agent_communication import AgentCommunicationHub, MessageType
from agents.domain_knowledge import DomainKnowledgeManager
from llm_models.specialized_llm import SpecializedLLM

class TestAgentGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test directories
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        
        cls.config_dir = cls.test_dir / "config"
        cls.config_dir.mkdir(exist_ok=True)
        
        cls.knowledge_dir = cls.test_dir / "knowledge"
        cls.knowledge_dir.mkdir(exist_ok=True)
        
    def setUp(self):
        """Set up each test."""
        self.comm_hub = AgentCommunicationHub()
        self.knowledge_manager = DomainKnowledgeManager(
            str(self.knowledge_dir)
        )
        
        self.factory = AgentFactory(
            self.comm_hub,
            self.knowledge_manager,
            str(self.config_dir)
        )
        
        self.generator = AgentGenerator(
            self.comm_hub,
            self.knowledge_manager,
            str(self.config_dir)
        )
        
    async def test_task_analysis(self):
        """Test task requirement analysis."""
        task = "Create a financial report and analyze market trends"
        
        # Mock LLM response
        mock_response = {
            "required_domains": ["finance", "market_analysis"],
            "capabilities": ["financial_reporting", "trend_analysis"],
            "knowledge_requirements": ["financial_metrics", "market_data"],
            "interaction_patterns": ["data_request", "report_generation"],
            "specialized_tools": ["data_visualization", "statistical_analysis"]
        }
        
        with patch.object(SpecializedLLM, 'generate_response',
                         return_value=json.dumps(mock_response)):
            requirements = await self.factory.analyze_task_requirements(task)
            
            self.assertEqual(
                requirements["required_domains"],
                ["finance", "market_analysis"]
            )
            self.assertIn("financial_reporting", requirements["capabilities"])
            
    async def test_agent_creation(self):
        """Test agent creation process."""
        spec = AgentSpecification(
            domain="test_domain",
            capabilities=["test_capability"],
            knowledge_requirements=["test_knowledge"],
            interaction_patterns=["test_pattern"],
            specialized_tools=["test_tool"],
            initial_prompts=["test prompt"],
            metadata={"test": "metadata"}
        )
        
        # Create agent
        agent = await self.factory.create_agent(spec)
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "test_domain")
        
        # Verify configuration saved
        config_file = self.config_dir / "test_domain_config.json"
        self.assertTrue(config_file.exists())
        
        with open(config_file) as f:
            config = json.load(f)
            self.assertEqual(config["domain"], "test_domain")
            
    async def test_agent_adaptation(self):
        """Test agent adaptation."""
        # Create initial agent
        spec = AgentSpecification(
            domain="adapt_test",
            capabilities=["initial_capability"],
            knowledge_requirements=["initial_knowledge"],
            interaction_patterns=["initial_pattern"],
            specialized_tools=["initial_tool"],
            initial_prompts=["initial prompt"],
            metadata={}
        )
        
        agent = await self.factory.create_agent(spec)
        
        # Test adaptation
        new_requirements = {
            "capabilities": ["new_capability"],
            "knowledge_requirements": ["new_knowledge"]
        }
        
        success = await self.factory.adapt_agent(
            "adapt_test",
            new_requirements
        )
        
        self.assertTrue(success)
        
        # Verify adaptation
        adapted_spec = self.factory.agent_specs["adapt_test"]
        self.assertIn("new_capability", adapted_spec.capabilities)
        
    async def test_agent_generator_request(self):
        """Test agent creation request handling."""
        task = "Analyze cryptocurrency market trends"
        
        # Mock task analysis response
        mock_analysis = {
            "required_domains": ["crypto", "market_analysis"],
            "capabilities": ["crypto_analysis", "trend_analysis"]
        }
        
        with patch.object(AgentFactory, 'analyze_task_requirements',
                         return_value=mock_analysis):
            response = await self.generator.handle_creation_request(
                task,
                "test_requester"
            )
            
            self.assertEqual(response["status"], "success")
            self.assertTrue(len(response["created_agents"]) > 0)
            
    async def test_performance_monitoring(self):
        """Test agent performance monitoring."""
        # Create test agent
        spec = AgentSpecification(
            domain="monitor_test",
            capabilities=["test_capability"],
            knowledge_requirements=["test_knowledge"],
            interaction_patterns=["test_pattern"],
            specialized_tools=["test_tool"],
            initial_prompts=["test prompt"],
            metadata={}
        )
        
        agent = await self.factory.create_agent(spec)
        
        # Mock performance metrics
        mock_performance = {
            "success_rate": 0.6,  # Below threshold
            "response_time": 2.0,
            "knowledge_coverage": 0.5
        }
        
        with patch.object(AgentGenerator, '_analyze_agent_performance',
                         return_value=mock_performance):
            # Run one monitoring cycle
            await self.generator.monitor_agent_performance()
            
            # Verify adaptation was triggered
            adapted_spec = self.factory.agent_specs["monitor_test"]
            self.assertNotEqual(
                adapted_spec.metadata.get("updated_at"),
                adapted_spec.metadata.get("created_at")
            )
            
    async def test_error_handling(self):
        """Test error handling in agent generation."""
        # Test invalid task
        with patch.object(AgentFactory, 'analyze_task_requirements',
                         side_effect=Exception("Analysis failed")):
            response = await self.generator.handle_creation_request(
                "invalid task",
                "test_requester"
            )
            
            self.assertEqual(response["status"], "failed")
            self.assertIn("error", response)
            
    async def test_concurrent_creation(self):
        """Test concurrent agent creation."""
        # Create multiple agents concurrently
        tasks = [
            "Analyze financial markets",
            "Develop marketing strategy",
            "Implement technical solution"
        ]
        
        async def create_request(task):
            return await self.generator.handle_creation_request(
                task,
                "test_requester"
            )
            
        # Run requests concurrently
        responses = await asyncio.gather(
            *[create_request(task) for task in tasks]
        )
        
        # Verify all requests completed
        self.assertEqual(len(responses), len(tasks))
        successful = [r for r in responses if r["status"] == "success"]
        self.assertEqual(len(successful), len(tasks))
        
    def test_configuration_persistence(self):
        """Test configuration persistence."""
        # Create test specification
        spec = AgentSpecification(
            domain="persist_test",
            capabilities=["test_capability"],
            knowledge_requirements=["test_knowledge"],
            interaction_patterns=["test_pattern"],
            specialized_tools=["test_tool"],
            initial_prompts=["test prompt"],
            metadata={}
        )
        
        # Save configuration
        self.factory._save_agent_configuration(spec)
        
        # Create new factory instance
        new_factory = AgentFactory(
            self.comm_hub,
            self.knowledge_manager,
            str(self.config_dir)
        )
        
        # Verify configuration loaded
        self.assertIn("persist_test", new_factory.agent_specs)
        loaded_spec = new_factory.agent_specs["persist_test"]
        self.assertEqual(loaded_spec.capabilities, spec.capabilities)
        
    def tearDown(self):
        """Clean up after each test."""
        # Clean up test files
        for file in self.config_dir.glob("*"):
            file.unlink()
        for file in self.knowledge_dir.glob("*"):
            file.unlink()
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test directories
        import shutil
        shutil.rmtree(cls.test_dir)

def async_test(coro):
    """Decorator for async test methods."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

# Apply async_test decorator to test methods
for name in dir(TestAgentGeneration):
    if name.startswith('test_'):
        attr = getattr(TestAgentGeneration, name)
        if asyncio.iscoroutinefunction(attr):
            setattr(TestAgentGeneration, name, async_test(attr))

if __name__ == '__main__':
    unittest.main()