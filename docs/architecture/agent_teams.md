# Agent Teams

Agent teams allow multiple agents to collaborate on shared goals. Each team is
stored under `teams/{id}.json` and represented by the `AgentTeam` dataclass.

```python
from core.teams import AgentTeam
team = AgentTeam.load("team1")
```

Members can join a team and take on roles like *lead* or *apprentice*. The team
specifies an optional `shared_goal` and a list of focus skills. Knowledge is
shared through the team knowledge module which writes JSON files per team.

