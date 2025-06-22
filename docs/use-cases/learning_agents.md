# Learning Agents

Agents may unlock new abilities by completing training paths. A critic agent can for example become a reviewer after finishing the corresponding path and receiving a positive evaluation from a mentor.

The training workflow:

1. Start a path with `agentnn training start <agent> --path <id>`
2. Complete the lessons or simulations
3. A coach submits an evaluation via `agentnn coach evaluate`
4. If the trust score is sufficient the skill is granted automatically
