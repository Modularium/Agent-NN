from training.reinforcement_learning import MultiAgentQLearner
from core.model_context import TaskContext


def test_team_learning_updates_values():
    learner = MultiAgentQLearner(learning_rate=1.0, epsilon=0.0, team_size=2)
    task = TaskContext(task_type="docker")
    learner.learn_team(task, ("a1", "a2"), reward=1.0)
    assert learner.table[("docker", ("a1", "a2"))] == 1.0


def test_select_team_returns_best():
    learner = MultiAgentQLearner(epsilon=0.0, team_size=2)
    task = TaskContext(task_type="docker")
    learner.table[("docker", ("a1", "a2"))] = 0.2
    learner.table[("docker", ("a1", "a3"))] = 0.8
    team = learner.select_team(task, ["a1", "a2", "a3"])
    assert team == ("a1", "a3")
