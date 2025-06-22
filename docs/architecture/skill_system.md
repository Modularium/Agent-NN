# Skill System

The skill system introduces declarative abilities that can be certified for each agent.
Skills are stored under `skills/{id}.json` and contain meta information like
required roles and expiration dates.

```
{
  "id": "text_evaluation",
  "title": "Text Evaluation",
  "required_for_roles": ["critic"],
  "expires_at": "2025-01-01T00:00:00Z"
}
```

Agent profiles keep a list of certified skills with timestamps:

```
{
  "id": "text_evaluation",
  "granted_at": "2024-01-01T12:00:00Z",
  "expires_at": "2025-01-01T00:00:00Z"
}
```

During dispatch the required skills can be specified in a `ModelContext`.
If an agent lacks a valid certification, a `missing_skills` warning is set and
audit logs record a `skill_check_failed` entry. When `enforce_certification`
is enabled, such tasks are skipped entirely.

