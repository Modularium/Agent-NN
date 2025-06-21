# Agent Profiles

Agent profiles store persistent information about every worker agent. They are saved as JSON files under `agent_profiles/` and loaded when an agent starts.

```python
from core.agent_profile import AgentIdentity
profile = AgentIdentity.load("writer_agent")
```

The `AgentIdentity` dataclass contains the role, traits and learned skills of an agent. Traits may influence prompts while skills give a rough overview of supported actions. After each successful task or feedback cycle the agent can update its profile:

```python
profile.traits["avg_deviation"] = 0.1
profile.save()
```

Profiles make it possible to track long term progress and allow other services to inspect an agent's current capabilities.
