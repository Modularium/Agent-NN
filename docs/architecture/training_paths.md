# Training Paths

Training paths define how an agent can acquire a new skill. Each path specifies prerequisites, a training method and which agent may certify the result.

The JSON files are stored under `training_paths/` and follow this structure:

```json
{
  "id": "review_flow",
  "target_skill": "reviewer",
  "prerequisites": ["critic"],
  "method": "task_simulation",
  "evaluation_prompt": "Please review the sample task result...",
  "certifier_agent": "mentor1",
  "mentor_required": true,
  "min_trust": 0.7
}
```

Agents start a path via the CLI and progress is tracked in their profile.
