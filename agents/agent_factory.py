"""Intelligent agent generation system."""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json
from pathlib import Path
import networkx as nx
from rich.console import Console
from rich.table import Table

from utils.logging_util import setup_logger
from llm_models.specialized_llm import SpecializedLLM
from llm_models.llm_backend import LLMBackendManager
from .worker_agent import WorkerAgent
from .agent_communication import AgentCommunicationHub, AgentMessage, MessageType
from .domain_knowledge import DomainKnowledgeManager

logger = setup_logger(__name__)
console = Console()

@dataclass
class AgentSpecification:
    """Specification for a new agent."""
    domain: str
    capabilities: List[str]
    knowledge_requirements: List[str]
    interaction_patterns: List[str]
    specialized_tools: List[str]
    initial_prompts: List[str]
    metadata: Dict[str, Any]

class AgentFactory:
    """Factory for creating specialized agents."""
    
    def __init__(self,
                 communication_hub: AgentCommunicationHub,
                 knowledge_manager: DomainKnowledgeManager,
                 config_dir: str = "config/agents"):
        """Initialize agent factory.
        
        Args:
            communication_hub: Communication hub for agents
            knowledge_manager: Domain knowledge manager
            config_dir: Directory for agent configurations
        """
        self.communication_hub = communication_hub
        self.knowledge_manager = knowledge_manager
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM for agent analysis
        self.llm = SpecializedLLM("agent_factory")
        
        # Track created agents
        self.agents: Dict[str, WorkerAgent] = {}
        self.agent_specs: Dict[str, AgentSpecification] = {}
        
        # Load existing configurations
        self._load_configurations()
        
    async def analyze_task_requirements(self, task_description: str) -> Dict[str, Any]:
        """Analyze task to determine agent requirements.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dict containing analysis results
        """
        # Prompt for task analysis
        prompt = f"""Analyze the following task and determine agent requirements:

Task: {task_description}

Please identify:
1. Required domains of expertise
2. Specific capabilities needed
3. Knowledge requirements
4. Interaction patterns
5. Specialized tools needed

Provide your analysis in JSON format with these fields."""

        try:
            response = self.llm.generate_response(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing task: {str(e)}")
            return {}
            
    async def determine_agent_needs(self,
                                  task_requirements: Dict[str, Any],
                                  existing_agents: List[str]) -> List[AgentSpecification]:
        """Determine what new agents are needed.
        
        Args:
            task_requirements: Requirements from task analysis
            existing_agents: List of existing agent names
            
        Returns:
            List of specifications for needed agents
        """
        needed_specs = []
        
        # Analyze existing capabilities
        existing_capabilities = await self._analyze_existing_capabilities(
            existing_agents
        )
        
        # Identify gaps
        for domain in task_requirements.get("required_domains", []):
            if not any(self._domain_covered(domain, existing_capabilities)):
                # Create specification for new agent
                spec = await self._create_agent_specification(
                    domain,
                    task_requirements
                )
                needed_specs.append(spec)
                
        return needed_specs
        
    async def create_agent(self, spec: AgentSpecification) -> Optional[WorkerAgent]:
        """Create a new agent from specification.
        
        Args:
            spec: Agent specification
            
        Returns:
            Created WorkerAgent or None if creation fails
        """
        try:
            # Initialize knowledge base
            initial_knowledge = await self._gather_domain_knowledge(spec)
            
            # Create agent
            agent = WorkerAgent(
                name=spec.domain,
                domain_docs=initial_knowledge,
                communication_hub=self.communication_hub,
                knowledge_manager=self.knowledge_manager
            )
            
            # Store agent and specification
            self.agents[spec.domain] = agent
            self.agent_specs[spec.domain] = spec
            
            # Save configuration
            self._save_agent_configuration(spec)
            
            # Start message processing
            asyncio.create_task(agent.process_messages())
            
            logger.info(f"Created new agent: {spec.domain}")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            return None
            
    async def adapt_agent(self,
                         agent_name: str,
                         new_requirements: Dict[str, Any]) -> bool:
        """Adapt an existing agent to new requirements.
        
        Args:
            agent_name: Name of agent to adapt
            new_requirements: New requirements
            
        Returns:
            bool: True if adaptation successful
        """
        if agent_name not in self.agents:
            return False
            
        try:
            agent = self.agents[agent_name]
            spec = self.agent_specs[agent_name]
            
            # Update specification
            new_spec = await self._update_specification(spec, new_requirements)
            
            # Gather additional knowledge
            new_knowledge = await self._gather_domain_knowledge(new_spec)
            
            # Update agent
            agent.ingest_knowledge(new_knowledge)
            self.agent_specs[agent_name] = new_spec
            
            # Save updated configuration
            self._save_agent_configuration(new_spec)
            
            logger.info(f"Adapted agent: {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adapting agent: {str(e)}")
            return False
            
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get capabilities of an agent.
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Dict containing agent capabilities
        """
        if agent_name not in self.agent_specs:
            return {}
            
        spec = self.agent_specs[agent_name]
        return {
            "domain": spec.domain,
            "capabilities": spec.capabilities,
            "knowledge_requirements": spec.knowledge_requirements,
            "interaction_patterns": spec.interaction_patterns,
            "specialized_tools": spec.specialized_tools
        }
        
    async def _analyze_existing_capabilities(self,
                                          agent_names: List[str]) -> Dict[str, Set[str]]:
        """Analyze capabilities of existing agents.
        
        Args:
            agent_names: List of agent names
            
        Returns:
            Dict mapping domains to capability sets
        """
        capabilities = {}
        for name in agent_names:
            if name in self.agent_specs:
                spec = self.agent_specs[name]
                capabilities[spec.domain] = set(spec.capabilities)
        return capabilities
        
    def _domain_covered(self,
                       domain: str,
                       existing_capabilities: Dict[str, Set[str]]) -> bool:
        """Check if a domain is covered by existing capabilities.
        
        Args:
            domain: Domain to check
            existing_capabilities: Existing agent capabilities
            
        Returns:
            bool: True if domain is covered
        """
        return domain in existing_capabilities
        
    async def _create_agent_specification(self,
                                        domain: str,
                                        task_requirements: Dict[str, Any]) -> AgentSpecification:
        """Create specification for a new agent.
        
        Args:
            domain: Agent domain
            task_requirements: Task requirements
            
        Returns:
            AgentSpecification for new agent
        """
        # Prompt for detailed specification
        prompt = f"""Create a detailed specification for a new agent:

Domain: {domain}
Task Requirements: {json.dumps(task_requirements, indent=2)}

Please specify:
1. Required capabilities
2. Knowledge requirements
3. Interaction patterns
4. Specialized tools needed
5. Initial prompts for the agent

Provide your specification in JSON format."""

        try:
            response = self.llm.generate_response(prompt)
            spec_data = json.loads(response)
            
            return AgentSpecification(
                domain=domain,
                capabilities=spec_data.get("capabilities", []),
                knowledge_requirements=spec_data.get("knowledge_requirements", []),
                interaction_patterns=spec_data.get("interaction_patterns", []),
                specialized_tools=spec_data.get("specialized_tools", []),
                initial_prompts=spec_data.get("initial_prompts", []),
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "task_requirements": task_requirements
                }
            )
        except Exception as e:
            logger.error(f"Error creating specification: {str(e)}")
            return AgentSpecification(
                domain=domain,
                capabilities=[],
                knowledge_requirements=[],
                interaction_patterns=[],
                specialized_tools=[],
                initial_prompts=[],
                metadata={}
            )
            
    async def _gather_domain_knowledge(self,
                                     spec: AgentSpecification) -> List[str]:
        """Gather knowledge for a new agent.
        
        Args:
            spec: Agent specification
            
        Returns:
            List of knowledge documents
        """
        knowledge = []
        
        # Get existing domain knowledge
        if self.knowledge_manager:
            domain_nodes = self.knowledge_manager.search_knowledge(
                spec.domain,
                limit=10
            )
            knowledge.extend([node.content for node in domain_nodes])
            
        # Generate initial knowledge from prompts
        for prompt in spec.initial_prompts:
            try:
                response = self.llm.generate_response(prompt)
                knowledge.append(response)
            except Exception as e:
                logger.error(f"Error generating knowledge: {str(e)}")
                
        return knowledge
        
    async def _update_specification(self,
                                  spec: AgentSpecification,
                                  new_requirements: Dict[str, Any]) -> AgentSpecification:
        """Update agent specification with new requirements.
        
        Args:
            spec: Existing specification
            new_requirements: New requirements
            
        Returns:
            Updated specification
        """
        # Prompt for specification update
        prompt = f"""Update agent specification with new requirements:

Current Specification:
{json.dumps(spec.__dict__, indent=2)}

New Requirements:
{json.dumps(new_requirements, indent=2)}

Please provide an updated specification in JSON format."""

        try:
            response = self.llm.generate_response(prompt)
            update_data = json.loads(response)
            
            return AgentSpecification(
                domain=spec.domain,
                capabilities=update_data.get("capabilities", spec.capabilities),
                knowledge_requirements=update_data.get(
                    "knowledge_requirements",
                    spec.knowledge_requirements
                ),
                interaction_patterns=update_data.get(
                    "interaction_patterns",
                    spec.interaction_patterns
                ),
                specialized_tools=update_data.get(
                    "specialized_tools",
                    spec.specialized_tools
                ),
                initial_prompts=update_data.get("initial_prompts", spec.initial_prompts),
                metadata={
                    **spec.metadata,
                    "updated_at": datetime.now().isoformat(),
                    "update_requirements": new_requirements
                }
            )
        except Exception as e:
            logger.error(f"Error updating specification: {str(e)}")
            return spec
            
    def _save_agent_configuration(self, spec: AgentSpecification):
        """Save agent configuration to disk.
        
        Args:
            spec: Agent specification
        """
        try:
            config_file = self.config_dir / f"{spec.domain}_config.json"
            with open(config_file, 'w') as f:
                json.dump(spec.__dict__, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            
    def _load_configurations(self):
        """Load existing agent configurations."""
        try:
            for config_file in self.config_dir.glob("*_config.json"):
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    spec = AgentSpecification(**data)
                    self.agent_specs[spec.domain] = spec
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            
    def show_agent_status(self):
        """Show status of all agents."""
        table = Table(title="Agent Status")
        table.add_column("Domain")
        table.add_column("Status")
        table.add_column("Capabilities")
        table.add_column("Last Updated")
        
        for domain, spec in self.agent_specs.items():
            status = "Active" if domain in self.agents else "Inactive"
            capabilities = ", ".join(spec.capabilities[:3])
            if len(spec.capabilities) > 3:
                capabilities += "..."
            last_updated = spec.metadata.get("updated_at",
                                          spec.metadata["created_at"])
            
            table.add_row(
                domain,
                status,
                capabilities,
                last_updated
            )
            
        console.print(table)