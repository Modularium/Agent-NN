# Dynamic Roles

Phase 3.5 introduces flexible role management. Roles form a hierarchy and can be temporarily extended.

## Role inheritance

`core/roles.py` defines `ROLE_HIERARCHY` mapping roles to inherited ones. `resolve_roles()` expands an agent's contract roles so that a `critic` automatically also owns the `reviewer` and `reader` privileges.

## Temporary roles

`AgentContract` may contain `temp_roles` that are valid for a single task. The dispatcher writes an audit entry when such a role is used and removes it from the contract afterwards. The applied roles are stored in `ModelContext.elevated_roles` for transparency.

## Trust based upgrades

`eligible_for_role()` in `core.trust_evaluator` checks if an agent qualifies for a new role based on the recorded trust score and standing. CLI commands allow verifying eligibility and performing an upgrade when appropriate.
