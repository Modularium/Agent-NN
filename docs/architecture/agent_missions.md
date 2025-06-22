# Agent Missions

Agent missions provide a structured, multi step objective for agents. Each mission
is stored as JSON under `missions/{id}.json` and loaded via `AgentMission`.
A mission defines steps with required skills, roles and deadlines. The dispatcher
annotates tasks with `mission_id`, `mission_step` and `mission_role` and updates
agent profiles automatically.
