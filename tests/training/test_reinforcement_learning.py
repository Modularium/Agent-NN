from training.reinforcement_learning import QTableLearner
from core.model_context import TaskContext


def test_qtable_learning_updates_values():
    learner = QTableLearner(learning_rate=1.0, epsilon=0.0)
    task = TaskContext(task_type="docker")
    learner.learn(task, "agent-a", reward=1.0)
    assert learner.table[("docker", "agent-a")] == 1.0


def test_qtable_select_agent_returns_best():
    learner = QTableLearner(epsilon=0.0)
    task = TaskContext(task_type="docker")
    learner.table[("docker", "a1")] = 0.2
    learner.table[("docker", "a2")] = 0.8
    agent = learner.select_agent(task, ["a1", "a2"])
    assert agent == "a2"
