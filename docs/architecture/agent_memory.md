# Agent Memory

The session manager can persist past interactions for every session.
Each entry contains the agent name, input, output and score.
A `memory` list is passed with every `ModelContext` so worker agents
can adapt their behaviour based on previous results.
