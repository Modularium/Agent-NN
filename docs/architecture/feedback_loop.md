# Feedback Loop Architecture

The feedback loop stores significant events for each agent to enable
self-improvement. Workers and services write `FeedbackLoopEntry` objects to
`feedback_loops/{agent_id}.jsonl` whenever tasks fail or negative ratings are
recorded.

During reflection the agent loads these entries and analyses patterns to adjust
its traits or skills. Adaptations are persisted in the agent profile and audited
via `adaptation_applied` actions.
