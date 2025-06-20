# Session Manager API

`POST /session`
: Create a new session with initial JSON data. Returns a generated `session_id`.

`GET /session/{id}`
: Retrieve data for a given session. If the session has expired, `null` is returned.

`GET /session/{id}/status`
: Returns `active` or `expired` depending on the session state.

`GET /health`
: Health check for the service.
