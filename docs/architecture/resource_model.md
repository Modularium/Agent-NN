# Agent Resource Model

The dispatcher now tracks economic metadata for each agent. `AgentIdentity` stores
estimated cost per token, average response time and a load factor between `0.0`
and `1.0`. The registry exposes these values so the dispatcher can select agents
based on the current load and expected expenses.

Load balancing works by filtering candidates with a `load_factor` below `0.8` and
sorting them by `estimated_cost_per_token` and `avg_response_time`. Workers
update their status via `/agent_status/{name}` which adjusts the persistent
profile.
