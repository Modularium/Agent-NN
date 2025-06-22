from core.coalitions import AgentCoalition
from services.coalition_manager.service import CoalitionManagerService


def test_create_and_assign(tmp_path, monkeypatch):
    monkeypatch.setenv("COALITION_DIR", str(tmp_path))
    service = CoalitionManagerService()
    coalition = service.create_coalition("demo goal", "leader", ["a", "b"], "parallel-expert")
    assert coalition.goal == "demo goal"
    coalition = service.assign_subtask(coalition.id, "write intro", "a")
    loaded = AgentCoalition.load(coalition.id)
    assert loaded.subtasks[0]["assigned_to"] == "a"
