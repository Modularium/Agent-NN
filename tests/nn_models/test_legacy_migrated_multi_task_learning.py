import pytest

pytest.importorskip("torch")


def test_import_multi_task_learning():
    from nn_models.multi_task_learning import TaskEncoder

    enc = TaskEncoder(input_dim=4, hidden_dims=[4], output_dim=2)
    assert hasattr(enc, 'forward')
