# Security Hardening

This document explains how to secure the Agent‑NN services.

## Token Authentication

Enable the middleware by setting `AUTH_ENABLED=true` in the environment. Valid bearer tokens are defined via `API_TOKENS` as a comma separated list. Rotate the tokens by updating the variable and restarting the services.

## Rate Limiting

Endpoints can be rate limited using the `slowapi` middleware. `RATE_LIMIT_TASK` controls the limits for the task dispatcher (default `10/minute`). Limits are only active when `RATE_LIMITS_ENABLED=true`.

When a client exceeds the limit a `429` response is returned and the `Retry-After` header indicates when the next request is allowed.

## Payload Protection

Input payloads are validated. Text inside `task_context.input_data.text` may contain at most 4096 characters. Invalid requests result in a 422 error.

## Deployment Tips

- Use a reverse proxy with TLS termination.
- Keep secrets such as API tokens outside of source control and load them via `.env` or your container orchestrator.
- Harden Docker images by using minimal base images and dropping privileges where possible.
