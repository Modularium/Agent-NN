# Adaptive Agents

With profiles each agent can record its own history and gradually adapt skills. A critic agent for example stores the average deviation of its ratings. Over time the system can analyse this metric and adjust the harshness or suggest new training data.

Profiles are accessible via the registry service:

```
GET /agent_profile/{name}
POST /agent_profile/{name}
```

This enables management tools to monitor progress and tweak agent behaviour during runtime.
