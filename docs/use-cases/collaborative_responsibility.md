# Collaborative Responsibility

Agents can delegate review or coordination duties to trusted peers. Example: A coordinator grants the reviewer role for a single task:

```bash
agentnn delegate grant coordinator_agent reviewer_agent --role reviewer --scope task --reason "sick leave"
```

When the reviewer submits the task, the dispatcher records `delegation_used` and attributes the result to the delegator.
