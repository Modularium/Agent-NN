import pytest

pytest.importorskip("torch")


def test_import_parallel_models():
    from nn_models.parallel_models import ParallelMode

    assert ParallelMode.TENSOR.value == 'tensor'
