from .registry import ToolRegistry
from .model_wrappers import MultiTaskModelWrapper, AgentV2Wrapper

ToolRegistry.register("multi_task_reasoner", MultiTaskModelWrapper)
ToolRegistry.register("agent_nn_v2", AgentV2Wrapper)

__all__ = ["ToolRegistry", "MultiTaskModelWrapper", "AgentV2Wrapper"]
