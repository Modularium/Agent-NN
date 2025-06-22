# Collective Learning

Teams of agents can distribute training efforts and share insights. After a
training run finishes, agents broadcast their results so others can improve
quickly.

Example CLI usage:

```bash
agentnn team create demo --coordinator alice
agentnn team join <team_id> --agent bob --role apprentice
agentnn team share-skill alice --skill review
```

