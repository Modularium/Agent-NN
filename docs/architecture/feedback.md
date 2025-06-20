# Feedback System

User feedback is stored in the session manager. Each feedback entry contains

- `index`: index of the message within the session history
- `rating`: `good` or `bad`
- `comment`: optional free text

Feedback can be submitted via `POST /chat/feedback`. The session manager
provides simple statistics via `GET /feedback/stats` returning total counts
of positive and negative ratings.
