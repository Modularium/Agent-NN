# Delegation Model

Delegations allow agents to temporarily or permanently pass a role to trusted peers. Each `DelegationGrant` is stored under `delegations/{delegator}.jsonl` and recorded in the profiles of both agents.

Only trusted peers may receive delegations. The dispatcher checks `has_valid_delegation` if an agent is not authorized for a role. On success the context contains `delegate_info` and an audit entry `delegation_used` is written.
