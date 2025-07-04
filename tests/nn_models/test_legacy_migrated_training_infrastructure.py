import pytest

pytest.importorskip("torch")


def test_import_training_infrastructure():
    from nn_models.training_infrastructure import GradientAccumulator

    assert GradientAccumulator is not None
