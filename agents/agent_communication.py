"""Agent communication and collaboration system."""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from utils.logging_util import setup_logger
from llm_models.llm_backend import LLMBackendManager, LLMBackendType
from datastores.worker_agent_db import WorkerAgentDB

logger = setup_logger(__name__)
console = Console()

class MessageType(Enum):
    """Types of messages that agents can exchange."""
    QUERY = "query"
    RESPONSE = "response"
    CLARIFICATION = "clarification"
    UPDATE = "update"
    ERROR = "error"
    TASK = "task"
    RESULT = "result"

@dataclass
class AgentMessage:
    """Message exchanged between agents."""
    message_type: MessageType
    sender: str
    receiver: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "type": self.message_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(
            message_type=MessageType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            metadata=data["metadata"],
            timestamp=data["timestamp"]
        )

class AgentCommunicationHub:
    """Central hub for agent communication."""
    
    def __init__(self, message_log_dir: str = "logs/messages"):
        """Initialize communication hub.
        
        Args:
            message_log_dir: Directory for message logs
        """
        self.message_log_dir = Path(message_log_dir)
        self.message_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Message queues for each agent
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # Knowledge sharing settings
        self.knowledge_sharing_enabled = True
        self.auto_update_enabled = True
        
        # Message history
        self.message_history: List[AgentMessage] = []
        
    async def register_agent(self, agent_name: str):
        """Register an agent with the hub.
        
        Args:
            agent_name: Name of the agent to register
        """
        if agent_name not in self.message_queues:
            self.message_queues[agent_name] = asyncio.Queue()
            logger.info(f"Registered agent: {agent_name}")
            
    async def deregister_agent(self, agent_name: str):
        """Deregister an agent from the hub.
        
        Args:
            agent_name: Name of the agent to deregister
        """
        if agent_name in self.message_queues:
            del self.message_queues[agent_name]
            logger.info(f"Deregistered agent: {agent_name}")
            
    async def send_message(self, message: AgentMessage):
        """Send a message to an agent.
        
        Args:
            message: Message to send
        """
        if message.receiver not in self.message_queues:
            logger.error(f"Unknown receiver: {message.receiver}")
            return
            
        await self.message_queues[message.receiver].put(message)
        self.message_history.append(message)
        
        # Log message
        self._log_message(message)
        
        # Handle knowledge sharing
        if self.knowledge_sharing_enabled:
            await self._handle_knowledge_sharing(message)
            
    async def get_messages(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of messages
        """
        if agent_name not in self.message_queues:
            return []
            
        messages = []
        while not self.message_queues[agent_name].empty():
            messages.append(await self.message_queues[agent_name].get())
            
        return messages
        
    async def broadcast_message(self,
                              sender: str,
                              content: str,
                              message_type: MessageType = MessageType.UPDATE,
                              metadata: Dict[str, Any] = None):
        """Broadcast a message to all agents.
        
        Args:
            sender: Name of sending agent
            content: Message content
            message_type: Type of message
            metadata: Optional message metadata
        """
        for receiver in self.message_queues.keys():
            if receiver != sender:
                message = AgentMessage(
                    message_type=message_type,
                    sender=sender,
                    receiver=receiver,
                    content=content,
                    metadata=metadata or {}
                )
                await self.send_message(message)
                
    async def request_clarification(self,
                                  sender: str,
                                  receiver: str,
                                  query: str,
                                  context: Dict[str, Any] = None):
        """Request clarification from another agent.
        
        Args:
            sender: Name of requesting agent
            receiver: Name of agent to ask
            query: Clarification query
            context: Optional context information
        """
        message = AgentMessage(
            message_type=MessageType.CLARIFICATION,
            sender=sender,
            receiver=receiver,
            content=query,
            metadata={"context": context or {}}
        )
        await self.send_message(message)
        
    async def delegate_task(self,
                          sender: str,
                          receiver: str,
                          task: str,
                          requirements: Dict[str, Any] = None):
        """Delegate a task to another agent.
        
        Args:
            sender: Name of delegating agent
            receiver: Name of agent to delegate to
            task: Task description
            requirements: Optional task requirements
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=sender,
            receiver=receiver,
            content=task,
            metadata={"requirements": requirements or {}}
        )
        await self.send_message(message)
        
    def get_conversation_history(self,
                               agent1: str,
                               agent2: str) -> List[AgentMessage]:
        """Get conversation history between two agents.
        
        Args:
            agent1: First agent name
            agent2: Second agent name
            
        Returns:
            List of messages between the agents
        """
        return [
            msg for msg in self.message_history
            if (msg.sender == agent1 and msg.receiver == agent2) or
               (msg.sender == agent2 and msg.receiver == agent1)
        ]
        
    def show_message_stats(self):
        """Show message statistics."""
        # Count messages by type
        type_counts = {}
        agent_counts = {}
        
        for msg in self.message_history:
            # Count by type
            msg_type = msg.message_type.value
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
            
            # Count by agent
            agent_counts[msg.sender] = agent_counts.get(msg.sender, 0) + 1
            agent_counts[msg.receiver] = agent_counts.get(msg.receiver, 0) + 1
            
        # Display statistics
        console.print("\n[bold]Message Statistics[/bold]")
        
        # Message types table
        type_table = Table(title="Messages by Type")
        type_table.add_column("Type")
        type_table.add_column("Count")
        
        for msg_type, count in type_counts.items():
            type_table.add_row(msg_type, str(count))
            
        console.print(type_table)
        
        # Agent activity table
        agent_table = Table(title="Agent Activity")
        agent_table.add_column("Agent")
        agent_table.add_column("Messages")
        
        for agent, count in agent_counts.items():
            agent_table.add_row(agent, str(count))
            
        console.print(agent_table)
        
    def _log_message(self, message: AgentMessage):
        """Log a message to disk.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.message_log_dir / f"messages_{timestamp}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(message.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error logging message: {str(e)}")
            
    async def _handle_knowledge_sharing(self, message: AgentMessage):
        """Handle knowledge sharing between agents.
        
        Args:
            message: Message that might contain shared knowledge
        """
        # Only share certain types of messages
        if message.message_type not in [
            MessageType.RESPONSE,
            MessageType.UPDATE,
            MessageType.RESULT
        ]:
            return
            
        # Check if message contains shareable knowledge
        if "shareable_knowledge" not in message.metadata:
            return
            
        knowledge = message.metadata["shareable_knowledge"]
        
        # Update knowledge bases of relevant agents
        if self.auto_update_enabled:
            try:
                receiver_db = WorkerAgentDB(message.receiver)
                receiver_db.ingest_documents(
                    [knowledge],
                    {"source": f"shared_by_{message.sender}"}
                )
                logger.info(f"Updated knowledge base of {message.receiver}")
            except Exception as e:
                logger.error(f"Error updating knowledge base: {str(e)}")
                
    def enable_knowledge_sharing(self, enabled: bool = True):
        """Enable or disable automatic knowledge sharing.
        
        Args:
            enabled: Whether to enable knowledge sharing
        """
        self.knowledge_sharing_enabled = enabled
        logger.info(f"Knowledge sharing {'enabled' if enabled else 'disabled'}")
        
    def enable_auto_update(self, enabled: bool = True):
        """Enable or disable automatic knowledge base updates.
        
        Args:
            enabled: Whether to enable auto updates
        """
        self.auto_update_enabled = enabled
        logger.info(f"Auto updates {'enabled' if enabled else 'disabled'}")