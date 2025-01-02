from typing import Dict, Any, Optional, List, Union
import asyncio
import json
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from utils.logging_util import LoggerMixin

class MessageType(Enum):
    """Types of inter-agent messages."""
    QUERY = "query"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    BROADCAST = "broadcast"

@dataclass
class AgentMessage:
    """Message for inter-agent communication."""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    priority: int = 1
    ttl: int = 300  # Time to live in seconds

class CommunicationManager(LoggerMixin):
    """Manager for inter-agent communication."""
    
    def __init__(self,
                 max_queue_size: int = 1000,
                 default_timeout: float = 30.0):
        """Initialize communication manager.
        
        Args:
            max_queue_size: Maximum message queue size
            default_timeout: Default message timeout in seconds
        """
        super().__init__()
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        
        # Message queues per agent
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # Active conversations
        self.conversations: Dict[str, Dict[str, Any]] = {}
        
        # Message history
        self.message_history: List[AgentMessage] = []
        
        # Registered agents
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self,
                      agent_id: str,
                      capabilities: List[str],
                      metadata: Optional[Dict[str, Any]] = None):
        """Register agent with communication system.
        
        Args:
            agent_id: Agent identifier
            capabilities: Agent capabilities
            metadata: Optional agent metadata
        """
        if agent_id in self.registered_agents:
            raise ValueError(f"Agent already registered: {agent_id}")
            
        # Create message queue
        self.message_queues[agent_id] = asyncio.Queue(
            maxsize=self.max_queue_size
        )
        
        # Store registration
        self.registered_agents[agent_id] = {
            "capabilities": capabilities,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
        
        # Log registration
        self.log_event(
            "agent_registered",
            {
                "agent_id": agent_id,
                "capabilities": capabilities
            }
        )
        
    def unregister_agent(self, agent_id: str):
        """Unregister agent from communication system.
        
        Args:
            agent_id: Agent identifier
        """
        if agent_id not in self.registered_agents:
            raise ValueError(f"Agent not registered: {agent_id}")
            
        # Remove queue
        del self.message_queues[agent_id]
        
        # Remove registration
        del self.registered_agents[agent_id]
        
        # Log unregistration
        self.log_event(
            "agent_unregistered",
            {"agent_id": agent_id}
        )
        
    async def send_message(self,
                          sender: str,
                          recipient: str,
                          content: Dict[str, Any],
                          message_type: MessageType = MessageType.QUERY,
                          correlation_id: Optional[str] = None,
                          priority: int = 1) -> str:
        """Send message to agent.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            content: Message content
            message_type: Message type
            correlation_id: Optional correlation ID
            priority: Message priority (1-10)
            
        Returns:
            str: Message ID
        """
        if recipient not in self.registered_agents:
            raise ValueError(f"Recipient not registered: {recipient}")
            
        # Create message
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender=sender,
            recipient=recipient,
            content=content,
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id,
            priority=priority
        )
        
        # Add to queue
        queue = self.message_queues[recipient]
        try:
            await queue.put(message)
        except asyncio.QueueFull:
            raise RuntimeError(f"Message queue full for agent: {recipient}")
            
        # Update conversation
        if correlation_id:
            if correlation_id not in self.conversations:
                self.conversations[correlation_id] = {
                    "started_at": datetime.now().isoformat(),
                    "messages": []
                }
            self.conversations[correlation_id]["messages"].append(
                message.message_id
            )
            
        # Add to history
        self.message_history.append(message)
        
        # Log message
        self.log_event(
            "message_sent",
            {
                "message_id": message.message_id,
                "sender": sender,
                "recipient": recipient,
                "type": message_type.value
            }
        )
        
        return message.message_id
        
    async def receive_message(self,
                            agent_id: str,
                            timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive message for agent.
        
        Args:
            agent_id: Agent identifier
            timeout: Optional timeout in seconds
            
        Returns:
            Optional[AgentMessage]: Received message or None if timeout
        """
        if agent_id not in self.registered_agents:
            raise ValueError(f"Agent not registered: {agent_id}")
            
        # Update last seen
        self.registered_agents[agent_id]["last_seen"] = datetime.now().isoformat()
        
        # Get from queue
        try:
            message = await asyncio.wait_for(
                self.message_queues[agent_id].get(),
                timeout=timeout or self.default_timeout
            )
            
            # Log receipt
            self.log_event(
                "message_received",
                {
                    "message_id": message.message_id,
                    "recipient": agent_id
                }
            )
            
            return message
            
        except asyncio.TimeoutError:
            return None
            
    async def broadcast_message(self,
                              sender: str,
                              content: Dict[str, Any],
                              recipients: Optional[List[str]] = None,
                              priority: int = 1) -> List[str]:
        """Broadcast message to multiple agents.
        
        Args:
            sender: Sender agent ID
            content: Message content
            recipients: Optional list of recipients (None for all)
            priority: Message priority
            
        Returns:
            List[str]: List of message IDs
        """
        # Get recipients
        if recipients is None:
            recipients = list(self.registered_agents.keys())
            
        # Send to each recipient
        message_ids = []
        for recipient in recipients:
            if recipient != sender:  # Don't send to self
                message_id = await self.send_message(
                    sender=sender,
                    recipient=recipient,
                    content=content,
                    message_type=MessageType.BROADCAST,
                    priority=priority
                )
                message_ids.append(message_id)
                
        return message_ids
        
    def get_conversation_history(self,
                               conversation_id: str) -> List[AgentMessage]:
        """Get conversation history.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List[AgentMessage]: Conversation messages
        """
        if conversation_id not in self.conversations:
            return []
            
        # Get message IDs
        message_ids = self.conversations[conversation_id]["messages"]
        
        # Get messages
        messages = []
        for message in self.message_history:
            if message.message_id in message_ids:
                messages.append(message)
                
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)
        return messages
        
    def get_agent_conversations(self, agent_id: str) -> List[str]:
        """Get agent's conversations.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List[str]: Conversation IDs
        """
        conversations = []
        for conv_id, conv in self.conversations.items():
            messages = conv["messages"]
            for message in self.message_history:
                if (message.message_id in messages and
                    (message.sender == agent_id or
                     message.recipient == agent_id)):
                    conversations.append(conv_id)
                    break
                    
        return conversations
        
    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get agent capabilities.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List[str]: Agent capabilities
        """
        if agent_id not in self.registered_agents:
            raise ValueError(f"Agent not registered: {agent_id}")
            
        return self.registered_agents[agent_id]["capabilities"]
        
    def find_capable_agents(self, capability: str) -> List[str]:
        """Find agents with specific capability.
        
        Args:
            capability: Required capability
            
        Returns:
            List[str]: Capable agent IDs
        """
        return [
            agent_id
            for agent_id, info in self.registered_agents.items()
            if capability in info["capabilities"]
        ]
        
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics.
        
        Returns:
            Dict[str, Any]: Communication statistics
        """
        stats = {
            "registered_agents": len(self.registered_agents),
            "active_conversations": len(self.conversations),
            "total_messages": len(self.message_history),
            "message_types": {},
            "agent_activity": {}
        }
        
        # Count message types
        for message in self.message_history:
            msg_type = message.message_type.value
            if msg_type not in stats["message_types"]:
                stats["message_types"][msg_type] = 0
            stats["message_types"][msg_type] += 1
            
        # Count agent activity
        for message in self.message_history:
            for agent_id in [message.sender, message.recipient]:
                if agent_id not in stats["agent_activity"]:
                    stats["agent_activity"][agent_id] = {
                        "sent": 0,
                        "received": 0
                    }
                    
            stats["agent_activity"][message.sender]["sent"] += 1
            stats["agent_activity"][message.recipient]["received"] += 1
            
        return stats
        
    async def cleanup_old_messages(self, max_age_hours: int = 24):
        """Clean up old messages and conversations.
        
        Args:
            max_age_hours: Maximum message age in hours
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=max_age_hours)
        
        # Clean messages
        self.message_history = [
            message for message in self.message_history
            if datetime.fromisoformat(message.timestamp) > cutoff
        ]
        
        # Clean conversations
        for conv_id in list(self.conversations.keys()):
            conv = self.conversations[conv_id]
            if datetime.fromisoformat(conv["started_at"]) <= cutoff:
                del self.conversations[conv_id]
                
        # Log cleanup
        self.log_event(
            "messages_cleaned",
            {"max_age_hours": max_age_hours}
        )