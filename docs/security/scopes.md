# API Scopes

Access to the gateway is controlled via scopes inside JWT tokens or associated with an API key.

| Scope | Description |
|-------|-------------|
| `llm:generate` | call the text generation endpoint |
| `chat:write` | send chat messages |
| `chat:read` | read chat history |
| `feedback:write` | send feedback |

Roles map to scopes as follows:

- **admin**: all scopes
- **agent**: `llm:generate`, `chat:write`, `chat:read`
- **client**: `chat:write`, `chat:read`
- **feedbacker**: `feedback:write`
