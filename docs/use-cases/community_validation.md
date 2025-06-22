# Community Validation

Some roles should be granted only after peer endorsement. Two or more agents can endorse a candidate using:

```bash
agentnn trust endorse mentor analyst --role reviewer --confidence 0.9
```

`agentnn trust circle analyst` returns a list of roles for which the agent is trusted. The dispatcher may require this status to assign sensitive missions.
