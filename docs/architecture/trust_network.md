# Trust Network

Agent recommendations create social trust circles. Each `AgentRecommendation` is stored under `recommendations/{agent}.jsonl` and updates the receiving profile.

`is_trusted_for(agent, role)` aggregates reputation and at least two peer endorsements with sufficient confidence (>=0.6). The dispatcher can require such endorsements before assigning a task.
