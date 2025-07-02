"""Training utilities."""

# Optional imports may pull in heavy dependencies. Wrap them in try/except so
# lightweight modules like :mod:`reinforcement_learning` can be used without
# requiring packages such as pandas or torch during import.
try:  # pragma: no cover - optional dependencies
    from .data_logger import (
        InteractionLogger,
        AgentInteractionDataset,
        create_dataloaders,
    )
    from .agent_selector_model import AgentSelectorModel, AgentSelectorTrainer
    from .federated import FederatedAveraging

    __all__ = [
        "InteractionLogger",
        "AgentInteractionDataset",
        "create_dataloaders",
        "AgentSelectorModel",
        "AgentSelectorTrainer",
        "FederatedAveraging",
    ]
except Exception:  # pragma: no cover - allow partial functionality
    __all__: list[str] = []
