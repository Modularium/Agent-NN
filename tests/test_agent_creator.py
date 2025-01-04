"""Tests for agent creator system."""
import pytest
import asyncio
import os
import tempfile
import shutil
import logging
from datetime import datetime
from typing import Dict, Any
import json
import yaml
from unittest.mock import patch, MagicMock
from rich.table import Table

logger = logging.getLogger(__name__)

from agents.agent_creator import AgentCreator
from agents.agent_communication import AgentCommunicationHub
from agents.domain_knowledge import DomainKnowledgeManager
from agents.worker_agent import WorkerAgent
from llm_models.specialized_llm import SpecializedLLM
from tests.test_mocks import (
    MockLLM,
    MockMLflow,
    MockWorkerAgent,
    MockCommunicationHub,
    MockKnowledgeManager,
    MockInteractionLogger,
    setup_mocks
)

@pytest.fixture
def mocks():
    """Set up all mocks for testing."""
    return setup_mocks()

@pytest.fixture
def mock_config():
    """Mock LLM config."""
    return {
        "llm": {
            "base_url": "http://192.168.0.247:1234",
            "chat_model": "llama-3.2-1b-instruct",
            "embedding_model": "text-embedding-nomic-embed-text-v1.5",
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None
        }
    }

@pytest.fixture
def agent_creator(mocks, mock_config) -> AgentCreator:
    """Agent creator for tests."""
    mock_file = MagicMock()
    mock_file.read.return_value = json.dumps(mock_config)
    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = None
    
    mock_open = MagicMock(return_value=mock_file)
    
    mock_llm = MockLLM()
    def mock_response(prompt: str) -> str:
        if "analyze" in prompt.lower() and "domain" in prompt.lower():
            return json.dumps(mock_llm.responses["analyze_domain"])
        elif "generate" in prompt.lower() and "knowledge" in prompt.lower():
            return json.dumps({
                "knowledge": mock_llm.responses["generate_knowledge"],
                "format": "list",
                "count": len(mock_llm.responses["generate_knowledge"])
            })
        return json.dumps({"response": "Mock response"})
    
    mock_llm.generate_response = MagicMock(side_effect=mock_response)
    
    with patch('agents.agent_creator.Path', return_value=mocks["path"]), \
         patch('builtins.open', mock_open), \
         patch('mlflow.start_run', mocks["mlflow"].start_run), \
         patch('mlflow.end_run', mocks["mlflow"].end_run), \
         patch('mlflow.log_param', mocks["mlflow"].log_param), \
         patch('mlflow.log_dict', mocks["mlflow"].log_dict), \
         patch('mlflow.log_metric', mocks["mlflow"].log_metric), \
         patch('mlflow.active_run', mocks["mlflow"].active_run), \
         patch('mlflow.set_experiment', mocks["mlflow"].set_experiment), \
         patch('mlflow.tracking.MlflowClient.get_experiment_by_name', mocks["mlflow"].get_experiment_by_name), \
         patch('agents.worker_agent.WorkerAgent', return_value=mocks["worker_agent"]), \
         patch('agents.agent_creator.InteractionLogger', return_value=mocks["interaction_logger"]), \
         patch('agents.agent_creator.SpecializedLLM', return_value=mock_llm):
        
        creator = AgentCreator(
            communication_hub=mocks["comm_hub"],
            knowledge_manager=mocks["knowledge_manager"],
            config_dir="/mock/config",
            performance_threshold=0.7
        )
        yield creator

@pytest.fixture
def sample_task() -> str:
    """Sample task description."""
    return "Build a web scraper to collect product information from e-commerce websites"

@pytest.fixture
def sample_domain_analysis() -> Dict[str, Any]:
    """Sample domain analysis result."""
    return {
        "primary_domain": "web_scraping",
        "capabilities": ["http_requests", "html_parsing", "data_extraction"],
        "knowledge_requirements": ["web_protocols", "html_structure", "data_formats"],
        "tools": ["requests", "beautifulsoup4", "selenium"],
        "metrics": ["success_rate", "data_accuracy", "scraping_speed"]
    }

@pytest.mark.asyncio
async def test_analyze_domain(agent_creator, sample_task):
    """Test domain analysis."""
    analysis = await agent_creator.analyze_domain(sample_task)
    
    assert isinstance(analysis, dict)
    assert analysis["primary_domain"] == "test_domain"
    assert analysis["capabilities"] == ["test_capability"]
    assert analysis["knowledge_requirements"] == ["test_knowledge"]
    assert analysis["tools"] == ["test_tool"]
    assert analysis["metrics"] == ["test_metric"]

@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_agent_unit(agent_creator, sample_task, mocks):
    """Test agent creation."""
    # Configure mock responses
    mocks["path"].exists.return_value = False
    mocks["open"].return_value.__enter__.return_value.read.return_value = "{}"
    
    # Configure mock LLM responses
    mock_llm = mocks["llm"]
    def mock_response(prompt: str) -> str:
        if "analyze" in prompt.lower() and "domain" in prompt.lower():
            return json.dumps(mock_llm.responses["analyze_domain"])
        elif "generate" in prompt.lower() and "knowledge" in prompt.lower():
            return json.dumps({
                "knowledge": mock_llm.responses["generate_knowledge"],
                "format": "list",
                "count": len(mock_llm.responses["generate_knowledge"])
            })
        return json.dumps({"response": "Mock response"})
    
    mock_llm.generate_response = MagicMock(side_effect=mock_response)
    
    # Configure mock worker agent
    mock_worker = mocks["worker_agent"]
    mock_worker.name = "test_domain_agent_20250103_103012"
    mock_worker.domain = "test_domain"
    
    try:
        logger.debug(f"Creating agent with task: {sample_task}")
        agent = await agent_creator.create_agent(sample_task)
        logger.debug(f"Created agent: {agent}")
        
        assert agent is not None
        assert isinstance(agent, MockWorkerAgent)
        assert agent.name in agent_creator.active_agents
        assert agent.name in agent_creator.agent_configs
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        logger.error(f"Mock LLM responses: {mock_llm.responses}")
        raise

@pytest.mark.asyncio
@pytest.mark.integration
async def test_create_agent_integration(agent_creator, sample_task):
    """Integration test for agent creation using LM-Studio."""
    agent = await agent_creator.create_agent(sample_task)
    
    assert agent is not None
    assert isinstance(agent, WorkerAgent)
    assert agent.name in agent_creator.active_agents
    assert agent.name in agent_creator.agent_configs
    
    # Test knowledge generation
    knowledge = agent.domain_docs
    assert len(knowledge) > 0
    
    # Test agent capabilities
    response = await agent.process_message("What are your capabilities?")
    assert response is not None
    assert len(response) > 0

@pytest.mark.asyncio
async def test_improve_agent(agent_creator, sample_task, mocks):
    """Test agent improvement."""
    # Create agent first
    agent = await agent_creator.create_agent(sample_task)
    assert agent is not None
    
    # Test improvement
    performance_data = {
        "success_rate": 0.6,
        "avg_response_time": 2.5,
        "avg_confidence": 0.65
    }
    
    result = await agent_creator.improve_agent(agent.name, performance_data)
    assert result is True

@pytest.mark.asyncio
async def test_monitor_performance(agent_creator, sample_task):
    """Test performance monitoring."""
    # Create agent
    agent = await agent_creator.create_agent(sample_task)
    assert agent is not None
    
    # Start monitoring task
    monitor_task = asyncio.create_task(agent_creator.monitor_performance())
    
    # Let it run for a bit
    await asyncio.sleep(1)
    
    # Cancel monitoring
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_needs_improvement(agent_creator):
    """Test improvement need detection."""
    # Test below threshold
    performance = {
        "success_rate": 0.6,
        "avg_confidence": 0.65
    }
    assert agent_creator._needs_improvement(performance) is True
    
    # Test above threshold
    performance = {
        "success_rate": 0.8,
        "avg_confidence": 0.85
    }
    assert agent_creator._needs_improvement(performance) is False

@pytest.mark.asyncio
async def test_get_agent_performance(agent_creator, sample_task, mocks):
    """Test getting agent performance."""
    # Create agent
    agent = await agent_creator.create_agent(sample_task)
    assert agent is not None
    
    # Configure mock interactions
    mocks["path"].glob.return_value = ["/mock/interactions/1.json"]
    mocks["open"].return_value.__enter__.return_value.read.return_value = json.dumps({
        "chosen_agent": agent.name,
        "success": True,
        "metrics": {
            "response_time": 0.5,
            "confidence": 0.8
        }
    })
    
    # Get performance
    performance = await agent_creator._get_agent_performance(agent.name)
    assert isinstance(performance, dict)
    assert "success_rate" in performance
    assert "avg_response_time" in performance
    assert "avg_confidence" in performance

@pytest.mark.asyncio
async def test_create_agent_config(agent_creator, sample_domain_analysis):
    """Test agent configuration creation."""
    config = await agent_creator._create_agent_config(
        domain="web_scraping",
        requirements=sample_domain_analysis
    )
    
    assert isinstance(config, dict)
    assert "name" in config
    assert "domain" in config
    assert "capabilities" in config
    assert "knowledge_requirements" in config
    assert "tools" in config
    assert "metrics" in config
    assert "created_at" in config
    assert "version" in config

@pytest.mark.asyncio
async def test_gather_domain_knowledge(agent_creator, sample_domain_analysis, mocks):
    """Test domain knowledge gathering."""
    # Configure mock knowledge search
    mocks["knowledge_manager"].search_knowledge.return_value = []
    
    knowledge = await agent_creator._gather_domain_knowledge(
        domain="web_scraping",
        requirements=sample_domain_analysis
    )
    
    assert isinstance(knowledge, list)
    assert len(knowledge) > 0

@pytest.mark.asyncio
async def test_analyze_improvement_needs(agent_creator, sample_task, mocks):
    """Test improvement needs analysis."""
    # Create agent first
    agent = await agent_creator.create_agent(sample_task)
    assert agent is not None
    
    # Configure mock configs
    mocks["open"].return_value.__enter__.return_value.read.return_value = json.dumps({
        "name": agent.name,
        "domain": "test_domain",
        "version": "1.0.0"
    })
    
    # Test analysis
    performance = {
        "success_rate": 0.6,
        "avg_response_time": 2.5,
        "avg_confidence": 0.65
    }
    
    analysis = await agent_creator._analyze_improvement_needs(
        agent.name,
        performance
    )
    
    assert isinstance(analysis, dict)
    assert "config_updates" in analysis
    assert "needs_knowledge" in analysis
    assert "knowledge_requirements" in analysis
    assert "needs_llm_update" in analysis
    assert "llm_updates" in analysis

@pytest.mark.asyncio
async def test_config_management(agent_creator, sample_task, mocks):
    """Test configuration management."""
    # Create agent to generate config
    agent = await agent_creator.create_agent(sample_task)
    assert agent is not None
    
    # Configure mock file operations
    config = {
        "name": agent.name,
        "domain": "test_domain",
        "version": "1.0.0"
    }
    mocks["open"].return_value.__enter__.return_value.read.return_value = json.dumps(config)
    mocks["path"].glob.return_value = ["/mock/config/test_config.json"]
    
    # Test config loading
    configs = agent_creator._load_configs()
    assert isinstance(configs, dict)
    assert agent.name in configs
    
    # Test config saving
    config["version"] = "1.0.1"
    agent_creator._save_config(config)
    
    # Verify update
    configs = agent_creator._load_configs()
    assert configs[agent.name]["version"] == "1.0.1"

def test_show_agent_status(agent_creator, capsys, mocks):
    """Test agent status display."""
    # Configure mock configs
    config = {
        "name": "test_agent",
        "domain": "test_domain",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat()
    }
    mocks["open"].return_value.__enter__.return_value.read.return_value = json.dumps(config)
    mocks["path"].glob.return_value = ["/mock/config/test_config.json"]
    
    # Load configs
    agent_creator._load_configs()
    
    # Show status
    agent_creator.show_agent_status()
    
    # Check output
    captured = capsys.readouterr()
    assert "Agent Status" in captured.out
    assert "test_agent" in captured.out
    assert "test_domain" in captured.out
    assert "1.0.0" in captured.out