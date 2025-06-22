# Adaptive Agent Behavior

Agents can modify their profile based on accumulated feedback. For example, a
critic agent reduces its assertiveness after several `task_failed` or
`criticism_received` events. The CLI provides `agentnn agent reflect` to preview
suggestions and `agentnn agent adapt` to apply them.
