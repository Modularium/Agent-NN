# Agent Evolution

Agents update their profile after processing tasks. The `evolve_profile` function merges feedback or LLM suggestions into the persistent `AgentIdentity`.

Modes:
- **llm** – use the prompt in `prompts/evolve_profile_prompt.txt` and send it to the LLM gateway. The reply must contain JSON with `traits` and `skills`.
- **heuristic** – adjust traits based on the ratio of good and bad ratings in the history.

When `AGENT_EVOLVE=true` a worker will call `evolve_profile` every few runs and store a log entry under `agent_log/`.
