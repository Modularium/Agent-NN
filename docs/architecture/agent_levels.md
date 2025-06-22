# Agent Levels

Agents advance through predefined levels to unlock new abilities. Each level is
defined in `levels/{id}.json` and contains required trust and skills.

Example definition:

```json
{
  "id": "analyst",
  "title": "Analyst",
  "trust_required": 0.6,
  "skills_required": ["research"],
  "unlocks": {"roles": ["analyst"], "rewards": {"tokens": 50}}
}
```

The `check_level_up` helper evaluates trust and certifications to decide when an
agent is promoted. Unlocks are applied to the agent contract and a
`level_up` entry is written to the audit log.
