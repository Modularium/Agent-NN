# Community Quality Feedback

After completing a mission agents can rate each other to highlight accuracy or collaboration skills.

```bash
agentnn rate mentor analyst --score 0.9 --tags critic,clarity
```

The command stores the rating in `ratings/analyst.jsonl` and updates the reputation score in the analyst's profile. Agents with high scores appear in the leaderboard:

```bash
agentnn rep leaderboard
```

This allows dispatchers to recommend reliable analysts or mentors based on community feedback.
