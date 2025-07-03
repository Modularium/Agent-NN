"""Unit tests for SupervisorAgent."""
import contextlib
import sys
import types
import pytest


class DummyAgent:
    def __init__(self, name="demo"):
        self.name = name

    def execute_task(self, task_description: str, context: str | None = None):
        return f"done:{task_description}"


class DummyAgentManager:
    def __init__(self):
        self.agents = {"demo_agent": DummyAgent("demo")}

    def get_all_agents(self):
        return list(self.agents.keys())

    def get_agent(self, name):
        return self.agents[name]

    def create_new_agent(self, task_description: str):
        agent = DummyAgent("dummy")
        new_name = f"{agent.name}_agent_{len(self.agents)+1}"
        self.agents[new_name] = agent
        return agent

    def get_agent_metadata(self, agent_name: str):
        return {"name": agent_name, "domain": "demo"}


class DummyNNManager:
    def __init__(self, return_agent: str | None = "demo_agent"):
        self.return_agent = return_agent

    def predict_best_agent(self, task_description: str, available_agents: list[str]):
        return self.return_agent

    def update_model(self, task_description: str, chosen_agent: str, success_score: float):
        pass


mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.start_run = lambda run_name=None: contextlib.nullcontext()
mlflow_stub.log_param = lambda *a, **k: None
mlflow_stub.log_metric = lambda *a, **k: None
sys.modules.setdefault("mlflow", mlflow_stub)

agent_manager_stub = types.ModuleType("managers.agent_manager")
agent_manager_stub.AgentManager = DummyAgentManager
sys.modules.setdefault("managers.agent_manager", agent_manager_stub)

nn_manager_stub = types.ModuleType("managers.nn_manager")
nn_manager_stub.NNManager = DummyNNManager
sys.modules.setdefault("managers.nn_manager", nn_manager_stub)

from agents.supervisor_agent import SupervisorAgent


def setup_supervisor(monkeypatch, chosen: str | None = "demo_agent") -> SupervisorAgent:
    monkeypatch.setattr("agents.supervisor_agent.AgentManager", DummyAgentManager)
    monkeypatch.setattr("agents.supervisor_agent.NNManager", lambda: DummyNNManager(chosen))

    # patch mlflow functions used in execute_task
    import agents.supervisor_agent as sup
    monkeypatch.setattr(sup.mlflow, "start_run", lambda run_name=None: contextlib.nullcontext())
    monkeypatch.setattr(sup.mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(sup.mlflow, "log_metric", lambda *a, **k: None)

    return SupervisorAgent()


@pytest.mark.unit
def test_execute_task_success(monkeypatch):
    sup = setup_supervisor(monkeypatch, "demo_agent")
    result = sup.execute_task("do something")
    assert result["success"] is True
    assert result["chosen_agent"] == "demo_agent"
    assert len(sup.task_history) == 1


@pytest.mark.unit
def test_execute_task_creates_new_agent(monkeypatch):
    sup = setup_supervisor(monkeypatch, None)
    result = sup.execute_task("another task")
    assert result["success"] is True
    assert result["chosen_agent"].startswith("dummy_agent_")
    assert len(sup.task_history) == 1


@pytest.mark.unit
def test_get_agent_status(monkeypatch):
    sup = setup_supervisor(monkeypatch, "demo_agent")
    sup.execute_task("t1")
    status = sup.get_agent_status("demo_agent")
    assert status["total_tasks"] == 1
    assert status["success_rate"] == 1.0


@pytest.mark.unit
def test_execution_history(monkeypatch):
    sup = setup_supervisor(monkeypatch, "demo_agent")
    sup.execute_task("task1")
    sup.execute_task("task2")
    history = sup.get_execution_history(1)
    assert len(history) == 1
    assert history[0]["task_description"] == "task2"


@pytest.mark.unit
def test_execute_task_failure(monkeypatch):
    def raise_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(DummyAgent, "execute_task", raise_error)
    sup = setup_supervisor(monkeypatch, "demo_agent")
    result = sup.execute_task("fail")
    assert result["success"] is False
    assert result["error"] == "boom"
    assert len(sup.task_history) == 1


@pytest.mark.unit
def test_get_status_without_history(monkeypatch):
    sup = setup_supervisor(monkeypatch, "demo_agent")
    status = sup.get_agent_status("demo_agent")
    assert status["total_tasks"] == 0
    assert status["success_rate"] == 0
    assert status["avg_execution_time"] == 0
    assert status["last_task_timestamp"] is None


@pytest.mark.unit
def test_update_model_called(monkeypatch):
    calls: list[float] = []

    def mock_update(self, task: str, agent: str, success_score: float):
        calls.append(success_score)

    sup = setup_supervisor(monkeypatch, "demo_agent")
    monkeypatch.setattr(SupervisorAgent, "_update_model", mock_update)
    sup.execute_task("ok")
    assert calls == [1.0]

    monkeypatch.setattr(DummyAgent, "execute_task", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad")))
    sup = setup_supervisor(monkeypatch, "demo_agent")
    monkeypatch.setattr(SupervisorAgent, "_update_model", mock_update)
    sup.execute_task("bad")
    assert calls[-1] == 0.0
