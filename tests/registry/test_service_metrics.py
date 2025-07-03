import pytest

from services.agent_registry.service import AgentRegistryService
from services.agent_registry.schemas import AgentInfo
from core.metrics_utils import TASKS_PROCESSED


@pytest.mark.unit
def test_service_metrics_increment():
    service = AgentRegistryService()
    info = AgentInfo(name="demo", url="http://demo")
    start = TASKS_PROCESSED.labels("agent_registry")._value.get()
    service.register_agent(info)
    after_register = TASKS_PROCESSED.labels("agent_registry")._value.get()
    assert after_register == start + 1

    service.list_agents()
    after_list = TASKS_PROCESSED.labels("agent_registry")._value.get()
    assert after_list == after_register + 1

    service.get_agent(info.id)
    after_get = TASKS_PROCESSED.labels("agent_registry")._value.get()
    assert after_get == after_list + 1


@pytest.mark.unit
def test_update_and_get_status():
    service = AgentRegistryService()
    service.update_status("agent", {"progress": 50})
    assert service.get_status("agent") == {"progress": 50}
