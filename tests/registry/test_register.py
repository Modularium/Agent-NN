from services.agent_registry.service import AgentRegistryService
from services.agent_registry.schemas import AgentInfo


def test_register_and_retrieve():
    service = AgentRegistryService()
    info = AgentInfo(name="agent", url="http://a", capabilities=["c1"])
    service.register_agent(info)
    assert service.get_agent(info.id) == info
    assert service.list_agents()[0] == info
