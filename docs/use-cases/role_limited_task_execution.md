# Role limited task execution

A critic agent is restricted to 1000 tokens. When the dispatcher forwards a task
with higher budget, it trims the limit and records a warning. The CLI command
`agentnn role limits critic` shows the configuration.
