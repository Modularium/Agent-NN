"""Specialized agent for generating new agents."""
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json
from pathlib import Path
from rich.console import Console

from utils.logging_util import setup_logger
from llm_models.specialized_llm import SpecializedLLM
from .worker_agent import WorkerAgent
from .agent_communication import AgentCommunicationHub, AgentMessage, MessageType
from .domain_knowledge import DomainKnowledgeManager
from .agent_factory import AgentFactory, AgentSpecification

logger = setup_logger(__name__)
console = Console()

class AgentGenerator(WorkerAgent):
    """Specialized agent for generating and managing other agents."""
    
    def __init__(self,
                 communication_hub: AgentCommunicationHub,
                 knowledge_manager: DomainKnowledgeManager,
                 config_dir: str = "config/agents"):
        """Initialize agent generator.
        
        Args:
            communication_hub: Communication hub for agents
            knowledge_manager: Domain knowledge manager
            config_dir: Directory for agent configurations
        """
        super().__init__(
            name="agent_generator",
            communication_hub=communication_hub,
            knowledge_manager=knowledge_manager
        )
        
        # Initialize agent factory
        self.factory = AgentFactory(
            communication_hub=communication_hub,
            knowledge_manager=knowledge_manager,
            config_dir=config_dir
        )
        
        # Track agent creation requests
        self.creation_requests: Dict[str, Dict[str, Any]] = {}
        
    async def handle_creation_request(self,
                                    task_description: str,
                                    requester: str) -> Dict[str, Any]:
        """Handle a request to create new agents.
        
        Args:
            task_description: Description of the task
            requester: Name of requesting agent
            
        Returns:
            Dict containing response about created agents
        """
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.creation_requests[request_id] = {
            "task": task_description,
            "requester": requester,
            "status": "analyzing",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Analyze task requirements
            requirements = await self.factory.analyze_task_requirements(
                task_description
            )
            
            # Get existing agents
            existing_agents = list(self.factory.agents.keys())
            
            # Determine needed agents
            needed_specs = await self.factory.determine_agent_needs(
                requirements,
                existing_agents
            )
            
            # Create new agents
            created_agents = []
            for spec in needed_specs:
                agent = await self.factory.create_agent(spec)
                if agent:
                    created_agents.append(spec.domain)
                    
            # Update request status
            self.creation_requests[request_id].update({
                "status": "completed",
                "created_agents": created_agents,
                "completion_time": datetime.now().isoformat()
            })
            
            # Notify requester
            await self._notify_requester(
                requester,
                request_id,
                created_agents
            )
            
            return {
                "request_id": request_id,
                "created_agents": created_agents,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error handling creation request: {str(e)}")
            self.creation_requests[request_id].update({
                "status": "failed",
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            })
            
            return {
                "request_id": request_id,
                "error": str(e),
                "status": "failed"
            }
            
    async def monitor_agent_performance(self):
        """Monitor performance of created agents."""
        while True:
            try:
                # Get all active agents
                active_agents = self.factory.agents
                
                for domain, agent in active_agents.items():
                    # Analyze recent performance
                    performance = await self._analyze_agent_performance(agent)
                    
                    # Check if adaptation needed
                    if self._needs_adaptation(performance):
                        # Generate adaptation requirements
                        requirements = await self._generate_adaptation_requirements(
                            domain,
                            performance
                        )
                        
                        # Adapt agent
                        success = await self.factory.adapt_agent(
                            domain,
                            requirements
                        )
                        
                        if success:
                            logger.info(f"Adapted agent: {domain}")
                        else:
                            logger.warning(f"Failed to adapt agent: {domain}")
                            
            except Exception as e:
                logger.error(f"Error monitoring agents: {str(e)}")
                
            await asyncio.sleep(300)  # Check every 5 minutes
            
    async def _analyze_agent_performance(self,
                                       agent: WorkerAgent) -> Dict[str, Any]:
        """Analyze an agent's performance.
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Dict containing performance metrics
        """
        # This is a placeholder for more sophisticated analysis
        return {
            "success_rate": 0.8,  # Example metric
            "response_time": 1.5,  # Example metric
            "knowledge_coverage": 0.7  # Example metric
        }
        
    def _needs_adaptation(self, performance: Dict[str, Any]) -> bool:
        """Determine if an agent needs adaptation.
        
        Args:
            performance: Performance metrics
            
        Returns:
            bool: True if adaptation needed
        """
        # This is a simple example - could be more sophisticated
        return (
            performance.get("success_rate", 1.0) < 0.7 or
            performance.get("knowledge_coverage", 1.0) < 0.6
        )
        
    async def _generate_adaptation_requirements(self,
                                             domain: str,
                                             performance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate requirements for agent adaptation.
        
        Args:
            domain: Agent domain
            performance: Performance metrics
            
        Returns:
            Dict containing adaptation requirements
        """
        # Prompt for adaptation requirements
        prompt = f"""Generate adaptation requirements for agent:

Domain: {domain}
Performance: {json.dumps(performance, indent=2)}

Please specify:
1. Areas needing improvement
2. Additional capabilities needed
3. Knowledge gaps to fill
4. Suggested adaptations

Provide requirements in JSON format."""

        try:
            response = self.llm.generate_response(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error generating requirements: {str(e)}")
            return {}
            
    async def _notify_requester(self,
                              requester: str,
                              request_id: str,
                              created_agents: List[str]):
        """Notify requesting agent about created agents.
        
        Args:
            requester: Name of requesting agent
            request_id: Request ID
            created_agents: List of created agent domains
        """
        if self.communication_hub:
            message = AgentMessage(
                message_type=MessageType.UPDATE,
                sender=self.name,
                receiver=requester,
                content=f"Created agents: {', '.join(created_agents)}",
                metadata={
                    "request_id": request_id,
                    "created_agents": created_agents
                }
            )
            await self.communication_hub.send_message(message)
            
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages.
        
        Args:
            message: Message to handle
        """
        try:
            if message.message_type == MessageType.TASK:
                # Check if it's an agent creation request
                if "create_agents" in message.metadata:
                    response = await self.handle_creation_request(
                        message.content,
                        message.sender
                    )
                    await self._send_response(
                        message.sender,
                        json.dumps(response),
                        MessageType.RESULT
                    )
                else:
                    # Handle other tasks normally
                    await super()._handle_message(message)
                    
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await self._send_response(
                message.sender,
                str(e),
                MessageType.ERROR
            )
            
    def show_creation_history(self):
        """Show history of agent creation requests."""
        table = Table(title="Agent Creation History")
        table.add_column("Request ID")
        table.add_column("Requester")
        table.add_column("Status")
        table.add_column("Created Agents")
        table.add_column("Timestamp")
        
        for req_id, req_data in self.creation_requests.items():
            created = ", ".join(
                req_data.get("created_agents", [])
            ) or "None"
            
            table.add_row(
                req_id,
                req_data["requester"],
                req_data["status"],
                created,
                req_data["timestamp"]
            )
            
        console.print(table)
        
    async def start(self):
        """Start the agent generator."""
        # Start message processing
        asyncio.create_task(self.process_messages())
        
        # Start performance monitoring
        asyncio.create_task(self.monitor_agent_performance())
        
        logger.info("Agent generator started")