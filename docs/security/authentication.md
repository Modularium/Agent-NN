# API Authentication

The API gateway can optionally enforce authentication. When enabled, clients must send an `X-API-Key` header with every request. The expected key is defined via the environment variable `API_GATEWAY_KEY`.

Set `API_AUTH_ENABLED=true` in `.env` to activate the check. When disabled, the gateway forwards requests without authentication.
