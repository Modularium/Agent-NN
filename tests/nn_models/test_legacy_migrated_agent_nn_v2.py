import pytest

pytest.importorskip("torch")


def test_import_agent_nn_v2():
    from nn_models.agent_nn_v2 import AgentNN

    assert AgentNN is not None
