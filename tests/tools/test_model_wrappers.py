from tools import ToolRegistry
from tools.model_wrappers import MultiTaskModelWrapper, AgentV2Wrapper


def test_registry_register_and_get():
    ToolRegistry._registry.clear()
    ToolRegistry.register("mtl", MultiTaskModelWrapper)
    ToolRegistry.register("v2", AgentV2Wrapper)
    assert set(ToolRegistry.list_tools()) == {"mtl", "v2"}
    tool = ToolRegistry.get("mtl")
    assert isinstance(tool, MultiTaskModelWrapper)
    assert isinstance(tool.run({}), dict)
