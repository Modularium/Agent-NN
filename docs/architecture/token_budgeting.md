# Token Budgeting

The dispatcher tracks token usage for each task and session. Every `ModelContext` carries optional fields `task_value`, `max_tokens` and `token_spent`. When a task exceeds its defined budget the dispatcher sets `warning` to `"budget exceeded"` and skips further processing. Token usage reported by worker agents is accumulated per session.
