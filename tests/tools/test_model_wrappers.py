from tools import ToolRegistry
from tools.model_wrappers import (
    AgentNNWrapper,
    AgentV2Wrapper,
    DynamicArchitectureWrapper,
    MultiTaskModelWrapper,
)


def test_registry_register_and_get():
    ToolRegistry._registry.clear()
    ToolRegistry.register("mtl", MultiTaskModelWrapper)
    ToolRegistry.register("v2", AgentV2Wrapper)
    ToolRegistry.register("dyn", DynamicArchitectureWrapper)
    ToolRegistry.register("base", AgentNNWrapper)
    assert set(ToolRegistry.list_tools()) == {"mtl", "v2", "dyn", "base"}
    tool = ToolRegistry.get("dyn")
    assert isinstance(tool, DynamicArchitectureWrapper)
    assert isinstance(tool.run({}), dict)
