# Distributed Teamwork Example

Teams of agents can now collaborate on a shared goal. After creating a coalition you can assign subtasks via the CLI:

```bash
agentnn team create --goal "Generate policy draft"
agentnn team assign <coalition_id> --to writer_agent --task "draft opening"
agentnn team status <coalition_id>
```

The dispatcher forwards tasks in `coalition` mode, the coordinator runs members in parallel and merges their results.
