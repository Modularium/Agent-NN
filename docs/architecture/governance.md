# Governance and Contracts

Phase 3 introduces a governance layer for Agent‑NN. Each agent can be bound to
an **AgentContract** stored under `contracts/`.

```python
@dataclass
class AgentContract:
    agent: str
    allowed_roles: List[str]
    max_tokens: int
    trust_level_required: float
    constraints: Dict[str, Any]
```

Before a task is assigned, the dispatcher loads the contract and checks the
required trust level using `calculate_trust`. If the agent role is not allowed or
the requested token budget exceeds `max_tokens`, the task is rejected and the
`ModelContext.warning` field is set.

Coalitions may also define a `ruleset` that enforces contracts and minimum trust
thresholds for all members.
