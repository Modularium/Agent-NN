# Agent Coalitions

Phase 2.6 introduces **agent coalitions** for distributed task solving. A coalition is stored as JSON under `coalitions/` and represented by `AgentCoalition`.

```python
@dataclass
class AgentCoalition:
    id: str
    goal: str
    leader: str
    members: List[str]
    strategy: str
    subtasks: List[Dict[str, Any]]
```

The `CoalitionManager` service manages creation and assignment. Strategies like `plan-then-split` or `parallel-expert` decide how subtasks are delegated. Each subtask status is tracked within the coalition file.
