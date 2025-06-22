# Role Capabilities

Each role now defines explicit resource limits. `core/role_capabilities.py` stores the mapping.
Dispatcher applies these limits before sending a task to an agent. The context records the
`applied_limits` so workers and clients know which caps were active.
