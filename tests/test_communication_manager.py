import unittest
from unittest.mock import patch, MagicMock
import asyncio
from datetime import datetime, timedelta
from managers.communication_manager import (
    CommunicationManager,
    MessageType,
    AgentMessage
)

class TestCommunicationManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Initialize manager
        self.manager = CommunicationManager(
            max_queue_size=10,
            default_timeout=1.0
        )
        
        # Register test agents
        self.manager.register_agent(
            "agent1",
            ["capability1", "capability2"]
        )
        self.manager.register_agent(
            "agent2",
            ["capability2", "capability3"]
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        
    async def test_send_receive_message(self):
        """Test message sending and receiving."""
        # Send message
        content = {"test": "data"}
        message_id = await self.manager.send_message(
            "agent1",
            "agent2",
            content
        )
        
        # Receive message
        message = await self.manager.receive_message("agent2")
        
        # Check message
        self.assertIsNotNone(message)
        self.assertEqual(message.message_id, message_id)
        self.assertEqual(message.sender, "agent1")
        self.assertEqual(message.recipient, "agent2")
        self.assertEqual(message.content, content)
        
    async def test_message_timeout(self):
        """Test message receive timeout."""
        # Try to receive (should timeout)
        message = await self.manager.receive_message(
            "agent1",
            timeout=0.1
        )
        
        # Check timeout
        self.assertIsNone(message)
        
    async def test_broadcast_message(self):
        """Test message broadcasting."""
        # Register more agents
        self.manager.register_agent(
            "agent3",
            ["capability1"]
        )
        
        # Broadcast message
        content = {"broadcast": "test"}
        message_ids = await self.manager.broadcast_message(
            "agent1",
            content
        )
        
        # Check messages sent
        self.assertEqual(len(message_ids), 2)  # Sent to agent2 and agent3
        
        # Receive messages
        message2 = await self.manager.receive_message("agent2")
        message3 = await self.manager.receive_message("agent3")
        
        # Check messages
        self.assertEqual(message2.content, content)
        self.assertEqual(message3.content, content)
        self.assertEqual(message2.message_type, MessageType.BROADCAST)
        self.assertEqual(message3.message_type, MessageType.BROADCAST)
        
    async def test_conversation_tracking(self):
        """Test conversation tracking."""
        # Start conversation
        conv_id = "test_conv"
        
        # Send messages
        await self.manager.send_message(
            "agent1",
            "agent2",
            {"msg": "1"},
            correlation_id=conv_id
        )
        await self.manager.send_message(
            "agent2",
            "agent1",
            {"msg": "2"},
            correlation_id=conv_id,
            message_type=MessageType.RESPONSE
        )
        
        # Get conversation history
        history = self.manager.get_conversation_history(conv_id)
        
        # Check history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].content["msg"], "1")
        self.assertEqual(history[1].content["msg"], "2")
        
    def test_agent_capabilities(self):
        """Test agent capability management."""
        # Check capabilities
        caps1 = self.manager.get_agent_capabilities("agent1")
        self.assertEqual(set(caps1), {"capability1", "capability2"})
        
        # Find capable agents
        agents = self.manager.find_capable_agents("capability2")
        self.assertEqual(set(agents), {"agent1", "agent2"})
        
    async def test_message_cleanup(self):
        """Test message cleanup."""
        # Send old message
        old_time = datetime.now() - timedelta(hours=48)
        message = AgentMessage(
            message_id="old",
            message_type=MessageType.QUERY,
            sender="agent1",
            recipient="agent2",
            content={"old": "message"},
            timestamp=old_time.isoformat()
        )
        self.manager.message_history.append(message)
        
        # Send new message
        await self.manager.send_message(
            "agent1",
            "agent2",
            {"new": "message"}
        )
        
        # Clean old messages
        await self.manager.cleanup_old_messages(max_age_hours=24)
        
        # Check cleanup
        self.assertEqual(len(self.manager.message_history), 1)
        self.assertEqual(
            self.manager.message_history[0].content,
            {"new": "message"}
        )
        
    def test_communication_stats(self):
        """Test communication statistics."""
        # Send test messages
        async def send_messages():
            await self.manager.send_message(
                "agent1",
                "agent2",
                {"test": 1}
            )
            await self.manager.send_message(
                "agent2",
                "agent1",
                {"test": 2},
                message_type=MessageType.RESPONSE
            )
            
        asyncio.run(send_messages())
        
        # Get stats
        stats = self.manager.get_communication_stats()
        
        # Check stats
        self.assertEqual(stats["registered_agents"], 2)
        self.assertEqual(stats["total_messages"], 2)
        self.assertEqual(
            stats["message_types"][MessageType.QUERY.value],
            1
        )
        self.assertEqual(
            stats["message_types"][MessageType.RESPONSE.value],
            1
        )
        
    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        # Register new agent
        self.manager.register_agent(
            "test_agent",
            ["test_capability"]
        )
        
        # Check registration
        self.assertIn("test_agent", self.manager.registered_agents)
        self.assertEqual(
            self.manager.registered_agents["test_agent"]["capabilities"],
            ["test_capability"]
        )
        
        # Unregister agent
        self.manager.unregister_agent("test_agent")
        
        # Check unregistration
        self.assertNotIn("test_agent", self.manager.registered_agents)
        self.assertNotIn("test_agent", self.manager.message_queues)

if __name__ == '__main__':
    unittest.main()