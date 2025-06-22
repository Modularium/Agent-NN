from core.coalitions import AgentCoalition
from services.coalition_manager.service import CoalitionManagerService


def test_task_split_and_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("COALITION_DIR", str(tmp_path))
    service = CoalitionManagerService()
    coal = service.create_coalition("goal", "leader", ["a"], "parallel-expert")
    service.assign_subtask(coal.id, "draft", "a")
    loaded = AgentCoalition.load(coal.id)
    assert loaded.subtasks[0]["title"] == "draft"
