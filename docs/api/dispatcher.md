# Task Dispatcher API

`POST /task`
: Accepts a task with `task_type`, `input` and optional `session_id`.
  The dispatcher looks up the session context, selects a worker via the
  Agent Registry and returns the worker response.

`GET /health`
: Simple health check returning `{"status": "ok"}`.
