# Voting Logic

The voting mode lets multiple worker agents propose answers which are then rated by one or more critic agents.

1. The dispatcher selects all matching agents including critics and sends them to the `AgentCoordinator` with `mode="voting"`.
2. Non critic agents generate their suggestions via the normal `/run` endpoint.
3. Each critic agent receives every suggestion through the new `/vote` endpoint and returns a `score` and optional `feedback`.
4. The coordinator aggregates the scores and stores them in `AgentRunContext.score` and `feedback`.
5. The suggestion with the highest average score becomes `aggregated_result` of the context.
