# Session Orchestrator

The session orchestrator manages multiple parallel sessions. It can
schedule new sessions with a priority value and pause or resume them on
request. The orchestrator exposes `schedule_session()`, `pause_session()`
and `resume_session()` methods.
