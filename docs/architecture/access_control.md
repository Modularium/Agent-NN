# Access Control Layer

Phase 3.4 adds a basic authorization layer to Agent‑NN. Each agent receives a role and every action is checked against a role based permission map. The dispatcher loads the agent contract and calls `is_authorized` before a task is assigned. If the role is not allowed the task is blocked and an audit entry is written.

The role list is defined in `core/roles.py` and currently includes writer, retriever, critic, analyst, reviewer and coordinator. Authorization checks are implemented in `core/access_control.py`.

Resources inside a `ModelContext` may define a list of roles that are allowed to read them via the `permissions` attribute. When the dispatcher prepares the context, fields are removed for agents that lack permission.
