"""Mock objects and utilities for testing."""
import json
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
from typing import Dict, Any, Optional, List

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Document

class MockLLM:
    """Mock LLM for testing."""
    def __init__(self):
        self.responses = {
            "analyze_domain": {
                "primary_domain": "test_domain",
                "capabilities": ["test_capability"],
                "knowledge_requirements": ["test_knowledge"],
                "tools": ["test_tool"],
                "metrics": ["test_metric"],
                "name": "test_agent",
                "domain": "test_domain",
                "version": "1.0.0"
            },
            "analyze_improvements": {
                "config_updates": {"version": "1.0.1"},
                "needs_knowledge": True,
                "knowledge_requirements": {"topics": ["test_topic"]},
                "needs_llm_update": False,
                "llm_updates": {}
            },
            "generate_knowledge": [
                "Test knowledge 1",
                "Test knowledge 2",
                "Test knowledge 3"
            ]
        }
    
    def generate_response(self, prompt: str) -> str:
        """Generate mock response based on prompt content."""
        try:
            if "analyze" in prompt.lower() and "domain" in prompt.lower():
                return json.dumps(self.responses["analyze_domain"])
            elif "analyze" in prompt.lower() and "improvement" in prompt.lower():
                return json.dumps(self.responses["analyze_improvements"])
            elif "generate" in prompt.lower() and "knowledge" in prompt.lower():
                return json.dumps({
                    "knowledge": self.responses["generate_knowledge"],
                    "format": "list",
                    "count": len(self.responses["generate_knowledge"])
                })
            return json.dumps({"response": "Mock response"})
        except Exception as e:
            logger.error(f"Error generating mock response: {str(e)}")
            return json.dumps({"error": str(e)})

class MockMLflow:
    """Mock MLflow for testing."""
    def __init__(self):
        self.active_runs = []
        self.metrics = {}
        self.params = {}
        self.artifacts = {}
        self.experiments = {}

    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Start a mock MLflow run."""
        if self.active_runs and not nested:
            raise Exception("Run already active")
        run_id = f"mock_run_{len(self.active_runs)}"
        run = MagicMock()
        run.info = MagicMock()
        run.info.run_id = run_id
        run.info.run_name = run_name
        run.info.status = "RUNNING"
        self.active_runs.append(run)
        return run

    def end_run(self):
        """End the current mock MLflow run."""
        if self.active_runs:
            run = self.active_runs.pop()
            run.info.status = "FINISHED"

    def log_metric(self, key: str, value: float):
        """Log a mock metric."""
        if not self.active_runs:
            raise Exception("No active run")
        run_id = self.active_runs[-1].info.run_id
        if run_id not in self.metrics:
            self.metrics[run_id] = {}
        self.metrics[run_id][key] = value

    def log_param(self, key: str, value: str):
        """Log a mock parameter."""
        if not self.active_runs:
            raise Exception("No active run")
        run_id = self.active_runs[-1].info.run_id
        if run_id not in self.params:
            self.params[run_id] = {}
        self.params[run_id][key] = value

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log a mock dictionary artifact."""
        if not self.active_runs:
            raise Exception("No active run")
        run_id = self.active_runs[-1].info.run_id
        if run_id not in self.artifacts:
            self.artifacts[run_id] = {}
        self.artifacts[run_id][artifact_file] = dictionary

    def active_run(self):
        """Get current active run."""
        return self.active_runs[-1] if self.active_runs else None

    def set_experiment(self, experiment_name: str):
        """Set the active experiment."""
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = {
                "name": experiment_name,
                "experiment_id": str(len(self.experiments)),
                "lifecycle_stage": "active",
                "creation_time": int(datetime.now().timestamp() * 1000),
                "last_update_time": int(datetime.now().timestamp() * 1000),
                "tags": {}
            }
        return self.experiments[experiment_name]

    def get_experiment_by_name(self, name: str):
        """Get experiment by name."""
        return self.experiments.get(name)

class MockWorkerAgent:
    """Mock WorkerAgent for testing."""
    def __init__(self, name: str = "test_agent", domain: str = "test_domain", domain_docs: Optional[List[str]] = None, config: Dict[str, Any] = None):
        self.name = name
        self.domain = domain
        self.config = config or {}
        self.domain_docs = domain_docs or []
        self.process_messages = AsyncMock()
        self.ingest_knowledge = MagicMock()
        self.nn = MagicMock()
        self.nn.load = MagicMock()
        self.nn.save = MagicMock()
        self.nn.train = MagicMock()
        self.nn.evaluate = MagicMock()
        self.nn.predict = MagicMock()
        self.nn.update = MagicMock()
        self.nn.get_metrics = MagicMock(return_value={"accuracy": 0.8, "loss": 0.2})
        self.nn.get_performance = MagicMock(return_value={"success_rate": 0.8, "avg_confidence": 0.85})
        self.nn.get_state = MagicMock(return_value={"weights": [], "biases": []})
        self.nn.set_state = MagicMock()
        self.nn.get_config = MagicMock(return_value={"layers": [], "activation": "relu"})
        self.nn.set_config = MagicMock()
        self.nn.get_history = MagicMock(return_value=[])

class MockCommunicationHub:
    """Mock AgentCommunicationHub for testing."""
    def __init__(self):
        self.agents = {}
        self.messages = []
        self.register_agent = AsyncMock()
        self.deregister_agent = AsyncMock()
        self.send_message = AsyncMock()
        self.broadcast_message = AsyncMock()

class MockKnowledgeManager:
    """Mock DomainKnowledgeManager for testing."""
    def __init__(self):
        self.knowledge = {}
        self.search_knowledge = MagicMock(return_value=[Document(page_content="Test knowledge", metadata={"domain": "test_domain"})])
        self.add_knowledge = MagicMock()
        self.get_knowledge = MagicMock()

class MockInteractionLogger:
    """Mock InteractionLogger for testing."""
    def __init__(self):
        self.interactions = []
        self.interactions_dir = MagicMock()
        self.log_interaction = MagicMock()

    def get_interactions(self, agent_name: str, start_time: Optional[datetime] = None):
        """Get mock interactions for an agent."""
        return [
            {
                "chosen_agent": agent_name,
                "success": True,
                "metrics": {
                    "response_time": 0.5,
                    "confidence": 0.8
                }
            }
        ]

def create_mock_file_system():
    """Create mock file system utilities."""
    # Mock Path
    mock_path = MagicMock()
    mock_path.exists = MagicMock(return_value=True)
    mock_path.mkdir = MagicMock()
    mock_path.glob = MagicMock(return_value=[])
    mock_path.__truediv__ = lambda self, other: mock_path  # Handle path / "subdir"
    mock_path.parent = mock_path
    mock_path.name = "mock_path"
    mock_path.is_file = MagicMock(return_value=True)
    mock_path.is_dir = MagicMock(return_value=True)
    mock_path.absolute = MagicMock(return_value=mock_path)
    mock_path.__str__ = lambda self: "/mock/path"
    
    # Mock file operations
    mock_file = MagicMock()
    mock_file.read = MagicMock(return_value="{}")
    mock_file.write = MagicMock()
    
    # Mock open context manager
    mock_open = MagicMock()
    mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
    mock_open.return_value.__exit__ = MagicMock()
    
    return mock_path, mock_open

def setup_mocks():
    """Set up all mocks for testing."""
    mocks = {
        "llm": MockLLM(),
        "mlflow": MockMLflow(),
        "worker_agent": MockWorkerAgent(),
        "comm_hub": MockCommunicationHub(),
        "knowledge_manager": MockKnowledgeManager(),
        "interaction_logger": MockInteractionLogger()
    }
    
    mock_path, mock_open = create_mock_file_system()
    mocks["path"] = mock_path
    mocks["open"] = mock_open
    
    return mocks