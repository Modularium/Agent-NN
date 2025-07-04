from .model_wrappers import (
    AgentNNWrapper,
    AgentV2Wrapper,
    DynamicArchitectureWrapper,
    MultiTaskModelWrapper,
)
from .registry import ToolRegistry

ToolRegistry.register("multi_task_reasoner", MultiTaskModelWrapper)
ToolRegistry.register("agent_nn_v2", AgentV2Wrapper)
ToolRegistry.register("dynamic_architecture", DynamicArchitectureWrapper)
ToolRegistry.register("agent_nn", AgentNNWrapper)

__all__ = [
    "ToolRegistry",
    "MultiTaskModelWrapper",
    "AgentV2Wrapper",
    "DynamicArchitectureWrapper",
    "AgentNNWrapper",
]
