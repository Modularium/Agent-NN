# Voting Use Case

A simple brainstorming session can make use of the voting mode.

```bash
curl -X POST http://dispatcher/task \
     -d '{"task_type": "demo", "mode": "voting", "description": "Propose taglines"}'
```

The dispatcher forwards the request to a writer agent and a critic agent. The critic rates each proposal and the coordinator returns the best one.
