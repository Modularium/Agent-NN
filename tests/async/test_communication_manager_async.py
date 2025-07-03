# ruff: noqa: E402
import sys
import types
import asyncio

mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.active_run = lambda: None
mlflow_stub.start_run = lambda *a, **kw: None
mlflow_stub.end_run = lambda: None
mlflow_stub.log_metrics = lambda *a, **kw: None
mlflow_stub.set_tags = lambda *a, **kw: None
sys.modules.setdefault("mlflow", mlflow_stub)

sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

import pytest

from managers.communication_manager import CommunicationManager


@pytest.mark.unit
def test_send_to_unregistered_agent():
    manager = CommunicationManager(max_queue_size=1, default_timeout=0.1)
    manager.register_agent("agent1", ["cap"])
    with pytest.raises(ValueError):
        asyncio.run(manager.send_message("agent1", "unknown", {"data": 1}))


@pytest.mark.unit
def test_queue_full_blocks():
    manager = CommunicationManager(max_queue_size=1, default_timeout=0.1)
    manager.register_agent("agent1", ["cap"])
    manager.register_agent("agent2", ["cap"])

    asyncio.run(manager.send_message("agent1", "agent2", {"data": 1}))
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            asyncio.wait_for(manager.send_message("agent1", "agent2", {"data": 2}), 0.1)
        )
