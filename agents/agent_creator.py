"""Automated agent creation and improvement system."""
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
from rich.console import Console

from utils.logging_util import setup_logger
from utils.prompts import create_combined_prompt
from llm_models.specialized_llm import SpecializedLLM
from llm_models.llm_backend import LLMBackendManager, LLMBackendType
from agents.worker_agent import WorkerAgent
from agents.agent_communication import AgentCommunicationHub
from agents.domain_knowledge import DomainKnowledgeManager
from training.data_logger import InteractionLogger

logger = setup_logger(__name__)
console = Console()

class AgentCreator:
    """System for creating and improving agents."""
    
    def __init__(self,
                 communication_hub: AgentCommunicationHub,
                 knowledge_manager: DomainKnowledgeManager,
                 config_dir: str = "config/agents",
                 performance_threshold: float = 0.7):
        """Initialize agent creator.
        
        Args:
            communication_hub: Communication hub
            knowledge_manager: Knowledge manager
            config_dir: Configuration directory
            performance_threshold: Performance threshold for improvement
        """
        self.communication_hub = communication_hub
        self.knowledge_manager = knowledge_manager
        self.config_dir = Path(config_dir)
        self.performance_threshold = performance_threshold
        
        # Initialize components
        self.llm = SpecializedLLM("agent_creator")
        self.backend_manager = LLMBackendManager()
        self.interaction_logger = InteractionLogger()
        
        # Load existing configurations
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.agent_configs = self._load_configs()
        
        # Track active agents
        self.active_agents: Dict[str, WorkerAgent] = {}
        
        # Start MLflow
        mlflow.set_experiment("agent_creation")
        
    async def analyze_domain(self, task_description: str) -> Dict[str, Any]:
        """Analyze task to determine domain requirements.
        
        Args:
            task_description: Task description
            
        Returns:
            Dict containing domain analysis
        """
        prompt = f"""Analyze the following task and determine domain requirements:

Task: {task_description}

Please identify:
1. Primary domain
2. Required capabilities
3. Knowledge requirements
4. Tools and APIs needed
5. Performance metrics to track

Provide your analysis in JSON format."""

        try:
            response = self.llm.generate_response(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing domain: {str(e)}")
            return {}
            
    async def create_agent(self,
                          task_description: str,
                          domain: Optional[str] = None) -> Optional[WorkerAgent]:
        """Create a new agent for a task.
        
        Args:
            task_description: Task description
            domain: Optional domain override
            
        Returns:
            Created WorkerAgent or None if creation fails
        """
        try:
            with mlflow.start_run(run_name="agent_creation"):
                # Log task
                mlflow.log_param("task_description", task_description)
                
                # Analyze domain requirements
                requirements = await self.analyze_domain(task_description)
                mlflow.log_dict(requirements, "domain_requirements.json")
                
                # Determine domain
                domain = domain or requirements.get("primary_domain")
                if not domain:
                    raise ValueError("Could not determine domain")
                    
                # Create agent configuration
                config = await self._create_agent_config(domain, requirements)
                mlflow.log_dict(config, "agent_config.json")
                
                # Initialize knowledge base
                knowledge = await self._gather_domain_knowledge(
                    domain,
                    requirements
                )
                
                # Create agent
                agent = WorkerAgent(
                    name=config["name"],
                    domain_docs=knowledge,
                    communication_hub=self.communication_hub,
                    knowledge_manager=self.knowledge_manager
                )
                
                # Store configuration
                self.agent_configs[config["name"]] = config
                self._save_config(config)
                
                # Add to active agents
                self.active_agents[config["name"]] = agent
                
                # Start message processing
                asyncio.create_task(agent.process_messages())
                
                # Log success
                mlflow.log_metric("creation_success", 1.0)
                logger.info(f"Created agent: {config['name']}")
                
                return agent
                
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            mlflow.log_metric("creation_success", 0.0)
            return None
            
    async def improve_agent(self,
                          agent_name: str,
                          performance_data: Dict[str, float]) -> bool:
        """Improve an existing agent.
        
        Args:
            agent_name: Name of agent to improve
            performance_data: Performance metrics
            
        Returns:
            bool: True if improvement successful
        """
        if agent_name not in self.active_agents:
            return False
            
        try:
            with mlflow.start_run(run_name="agent_improvement"):
                # Log current performance
                mlflow.log_metrics(performance_data)
                
                # Analyze improvement needs
                improvements = await self._analyze_improvement_needs(
                    agent_name,
                    performance_data
                )
                mlflow.log_dict(improvements, "improvement_needs.json")
                
                # Update configuration
                config = self.agent_configs[agent_name]
                config.update(improvements.get("config_updates", {}))
                
                # Gather additional knowledge
                if improvements.get("needs_knowledge", False):
                    new_knowledge = await self._gather_domain_knowledge(
                        config["domain"],
                        improvements.get("knowledge_requirements", {})
                    )
                    
                    # Add to agent
                    agent = self.active_agents[agent_name]
                    agent.ingest_knowledge(new_knowledge)
                    
                # Update LLM if needed
                if improvements.get("needs_llm_update", False):
                    await self._update_agent_llm(agent_name, improvements)
                    
                # Save updated configuration
                self._save_config(config)
                
                # Log success
                mlflow.log_metric("improvement_success", 1.0)
                logger.info(f"Improved agent: {agent_name}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error improving agent: {str(e)}")
            mlflow.log_metric("improvement_success", 0.0)
            return False
            
    async def monitor_performance(self):
        """Monitor agent performance and trigger improvements."""
        while True:
            try:
                for name, agent in self.active_agents.items():
                    # Get recent performance
                    performance = await self._get_agent_performance(name)
                    
                    # Check if improvement needed
                    if self._needs_improvement(performance):
                        await self.improve_agent(name, performance)
                        
            except Exception as e:
                logger.error(f"Error monitoring performance: {str(e)}")
                
            await asyncio.sleep(300)  # Check every 5 minutes
            
    def _needs_improvement(self, performance: Dict[str, float]) -> bool:
        """Check if agent needs improvement.
        
        Args:
            performance: Performance metrics
            
        Returns:
            bool: True if improvement needed
        """
        return any(
            metric < self.performance_threshold
            for metric in performance.values()
        )
        
    async def _get_agent_performance(self, agent_name: str) -> Dict[str, float]:
        """Get agent performance metrics.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Dict containing performance metrics
        """
        # Get recent interactions
        interactions = []
        for file in self.interaction_logger.interactions_dir.glob("*.json"):
            try:
                with open(file) as f:
                    interaction = json.load(f)
                    if interaction["chosen_agent"] == agent_name:
                        interactions.append(interaction)
            except Exception as e:
                logger.error(f"Error loading interaction: {str(e)}")
                
        if not interactions:
            return {}
            
        # Calculate metrics
        success_rate = np.mean([int(i["success"]) for i in interactions])
        
        metrics = {
            "success_rate": success_rate,
            "avg_response_time": np.mean([
                i["metrics"].get("response_time", 0)
                for i in interactions
            ]),
            "avg_confidence": np.mean([
                i["metrics"].get("confidence", 0)
                for i in interactions
            ])
        }
        
        return metrics
        
    async def _create_agent_config(self,
                                 domain: str,
                                 requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create agent configuration.
        
        Args:
            domain: Agent domain
            requirements: Domain requirements
            
        Returns:
            Dict containing agent configuration
        """
        # Generate unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{domain}_agent_{timestamp}"
        
        config = {
            "name": name,
            "domain": domain,
            "capabilities": requirements.get("capabilities", []),
            "knowledge_requirements": requirements.get("knowledge_requirements", []),
            "tools": requirements.get("tools", []),
            "metrics": requirements.get("metrics", []),
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        return config
        
    async def _gather_domain_knowledge(self,
                                     domain: str,
                                     requirements: Dict[str, Any]) -> List[str]:
        """Gather knowledge for domain.
        
        Args:
            domain: Domain name
            requirements: Domain requirements
            
        Returns:
            List of knowledge documents
        """
        knowledge = []
        
        # Get existing domain knowledge
        domain_docs = self.knowledge_manager.search_knowledge(domain)
        knowledge.extend([doc.content for doc in domain_docs])
        
        # Generate additional knowledge if needed
        if len(knowledge) < 5:  # Minimum knowledge threshold
            prompt = f"""Generate foundational knowledge for domain:

Domain: {domain}
Requirements: {json.dumps(requirements, indent=2)}

Generate 5 key concepts or principles for this domain."""

            try:
                response = self.llm.generate_response(prompt)
                knowledge.append(response)
            except Exception as e:
                logger.error(f"Error generating knowledge: {str(e)}")
                
        return knowledge
        
    async def _analyze_improvement_needs(self,
                                       agent_name: str,
                                       performance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze agent improvement needs.
        
        Args:
            agent_name: Agent name
            performance: Performance metrics
            
        Returns:
            Dict containing improvement requirements
        """
        config = self.agent_configs[agent_name]
        
        prompt = f"""Analyze agent performance and suggest improvements:

Agent Configuration:
{json.dumps(config, indent=2)}

Performance Metrics:
{json.dumps(performance, indent=2)}

Suggest improvements in JSON format with fields:
1. config_updates
2. needs_knowledge
3. knowledge_requirements
4. needs_llm_update
5. llm_updates"""

        try:
            response = self.llm.generate_response(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing improvements: {str(e)}")
            return {}
            
    async def _update_agent_llm(self,
                               agent_name: str,
                               improvements: Dict[str, Any]):
        """Update agent's LLM configuration.
        
        Args:
            agent_name: Agent name
            improvements: Improvement requirements
        """
        if agent_name not in self.active_agents:
            return
            
        agent = self.active_agents[agent_name]
        updates = improvements.get("llm_updates", {})
        
        try:
            # Update LLM configuration
            if "model" in updates:
                agent.llm = SpecializedLLM(
                    domain=agent.name,
                    model_name=updates["model"]
                )
                
            # Update prompts if needed
            if "prompts" in updates:
                new_prompt = create_combined_prompt(
                    agent.name,
                    additional_context=updates["prompts"]
                )
                agent.qa_chain.prompt = new_prompt
                
        except Exception as e:
            logger.error(f"Error updating LLM: {str(e)}")
            
    def _load_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load agent configurations."""
        configs = {}
        
        for file in self.config_dir.glob("*.json"):
            try:
                with open(file) as f:
                    config = json.load(f)
                    configs[config["name"]] = config
            except Exception as e:
                logger.error(f"Error loading config {file}: {str(e)}")
                
        return configs
        
    def _save_config(self, config: Dict[str, Any]):
        """Save agent configuration.
        
        Args:
            config: Agent configuration
        """
        try:
            file_path = self.config_dir / f"{config['name']}_config.json"
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            
    def show_agent_status(self):
        """Show status of all agents."""
        table = Table(title="Agent Status")
        table.add_column("Name")
        table.add_column("Domain")
        table.add_column("Status")
        table.add_column("Version")
        table.add_column("Last Updated")
        
        for name, config in self.agent_configs.items():
            status = "Active" if name in self.active_agents else "Inactive"
            
            table.add_row(
                name,
                config["domain"],
                status,
                config["version"],
                config.get("updated_at", config["created_at"])
            )
            
        console.print(table)