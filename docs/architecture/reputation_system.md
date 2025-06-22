# Reputation System

Peer ratings provide a social signal for agent quality. A rating is stored as JSON lines under `ratings/{agent}.jsonl` using the `AgentRating` dataclass. It captures the score, optional feedback and context tags.

The reputation score of an agent is the average of all ratings. Whenever a new rating is saved the profile of the target agent is updated with `reputation_score` and a `feedback_log` of the last ten entries.

The trust evaluator exposes `aggregate_reputation(agent_id)` to obtain the current score which can be used by the dispatcher for task matching or audits.
