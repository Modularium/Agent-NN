"""Tests for agent communication system."""
import pytest
import asyncio
import os
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any

from agents.agent_communication import (
    MessageType,
    AgentMessage,
    AgentCommunicationHub
)

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def message_data() -> Dict[str, Any]:
    """Sample message data for tests."""
    return {
        "type": "query",
        "sender": "agent1",
        "receiver": "agent2",
        "content": "test message",
        "metadata": {"key": "value"},
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def sample_message(message_data) -> AgentMessage:
    """Sample message for tests."""
    return AgentMessage.from_dict(message_data)

@pytest.fixture
def comm_hub(temp_dir) -> AgentCommunicationHub:
    """Communication hub for tests."""
    return AgentCommunicationHub(message_log_dir=temp_dir)

@pytest.mark.asyncio
async def test_message_conversion(message_data, sample_message):
    """Test message conversion between dict and object."""
    # Test from_dict
    assert sample_message.message_type == MessageType.QUERY
    assert sample_message.sender == "agent1"
    assert sample_message.receiver == "agent2"
    assert sample_message.content == "test message"
    assert sample_message.metadata == {"key": "value"}
    assert sample_message.timestamp == message_data["timestamp"]
    
    # Test to_dict
    converted = sample_message.to_dict()
    assert converted == message_data

@pytest.mark.asyncio
async def test_agent_registration(comm_hub):
    """Test agent registration and deregistration."""
    # Register agent
    await comm_hub.register_agent("test_agent")
    assert "test_agent" in comm_hub.message_queues
    
    # Deregister agent
    await comm_hub.deregister_agent("test_agent")
    assert "test_agent" not in comm_hub.message_queues

@pytest.mark.asyncio
async def test_message_sending(comm_hub, sample_message):
    """Test sending messages between agents."""
    # Register agents
    await comm_hub.register_agent("agent1")
    await comm_hub.register_agent("agent2")
    
    # Send message
    await comm_hub.send_message(sample_message)
    
    # Get messages
    messages = await comm_hub.get_messages("agent2")
    assert len(messages) == 1
    assert messages[0].content == "test message"

@pytest.mark.asyncio
async def test_broadcast_message(comm_hub):
    """Test broadcasting messages."""
    # Register agents
    await comm_hub.register_agent("agent1")
    await comm_hub.register_agent("agent2")
    await comm_hub.register_agent("agent3")
    
    # Broadcast message
    await comm_hub.broadcast_message(
        sender="agent1",
        content="broadcast test",
        message_type=MessageType.UPDATE
    )
    
    # Check messages
    messages2 = await comm_hub.get_messages("agent2")
    messages3 = await comm_hub.get_messages("agent3")
    
    assert len(messages2) == 1
    assert len(messages3) == 1
    assert messages2[0].content == "broadcast test"
    assert messages3[0].content == "broadcast test"

@pytest.mark.asyncio
async def test_clarification_request(comm_hub):
    """Test requesting clarification."""
    # Register agents
    await comm_hub.register_agent("agent1")
    await comm_hub.register_agent("agent2")
    
    # Request clarification
    await comm_hub.request_clarification(
        sender="agent1",
        receiver="agent2",
        query="please clarify",
        context={"topic": "test"}
    )
    
    # Check message
    messages = await comm_hub.get_messages("agent2")
    assert len(messages) == 1
    assert messages[0].message_type == MessageType.CLARIFICATION
    assert messages[0].content == "please clarify"
    assert messages[0].metadata["context"] == {"topic": "test"}

@pytest.mark.asyncio
async def test_task_delegation(comm_hub):
    """Test task delegation."""
    # Register agents
    await comm_hub.register_agent("agent1")
    await comm_hub.register_agent("agent2")
    
    # Delegate task
    await comm_hub.delegate_task(
        sender="agent1",
        receiver="agent2",
        task="test task",
        requirements={"priority": "high"}
    )
    
    # Check message
    messages = await comm_hub.get_messages("agent2")
    assert len(messages) == 1
    assert messages[0].message_type == MessageType.TASK
    assert messages[0].content == "test task"
    assert messages[0].metadata["requirements"] == {"priority": "high"}

@pytest.mark.asyncio
async def test_conversation_history(comm_hub):
    """Test getting conversation history."""
    # Register agents
    await comm_hub.register_agent("agent1")
    await comm_hub.register_agent("agent2")
    
    # Send messages
    message1 = AgentMessage(
        message_type=MessageType.QUERY,
        sender="agent1",
        receiver="agent2",
        content="message 1",
        metadata={}
    )
    message2 = AgentMessage(
        message_type=MessageType.RESPONSE,
        sender="agent2",
        receiver="agent1",
        content="message 2",
        metadata={}
    )
    
    await comm_hub.send_message(message1)
    await comm_hub.send_message(message2)
    
    # Get history
    history = comm_hub.get_conversation_history("agent1", "agent2")
    assert len(history) == 2
    assert history[0].content == "message 1"
    assert history[1].content == "message 2"

@pytest.mark.asyncio
async def test_knowledge_sharing(comm_hub):
    """Test knowledge sharing functionality."""
    # Register agents
    await comm_hub.register_agent("agent1")
    await comm_hub.register_agent("agent2")
    
    # Enable knowledge sharing
    comm_hub.enable_knowledge_sharing(True)
    comm_hub.enable_auto_update(True)
    
    # Send message with shareable knowledge
    message = AgentMessage(
        message_type=MessageType.UPDATE,
        sender="agent1",
        receiver="agent2",
        content="knowledge update",
        metadata={"shareable_knowledge": "test knowledge"}
    )
    
    await comm_hub.send_message(message)
    
    # Verify message was sent
    messages = await comm_hub.get_messages("agent2")
    assert len(messages) == 1
    assert messages[0].metadata["shareable_knowledge"] == "test knowledge"

@pytest.mark.asyncio
async def test_message_logging(comm_hub, sample_message):
    """Test message logging."""
    # Register agents
    await comm_hub.register_agent("agent1")
    await comm_hub.register_agent("agent2")
    
    # Send message
    await comm_hub.send_message(sample_message)
    
    # Check log file
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(comm_hub.message_log_dir, f"messages_{timestamp}.jsonl")
    
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        log_content = f.read().strip()
        assert "test message" in log_content