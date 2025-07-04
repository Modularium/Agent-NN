import pytest

pytest.importorskip("torch")


def test_import_distributed_training():
    from nn_models.distributed_training import DistributedTrainer

    assert DistributedTrainer is not None
