# Contract enforced execution

Sensitive tasks may only run on agents that signed an explicit contract. The
dispatcher verifies `allowed_roles` and `trust_level_required` before invoking a
worker. If no agent meets the governance requirements, the task is refused.
