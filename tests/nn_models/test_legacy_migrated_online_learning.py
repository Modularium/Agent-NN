import pytest

pytest.importorskip("torch")


def test_import_online_learning():
    from nn_models.online_learning import StreamingBuffer

    buf = StreamingBuffer(capacity=10, feature_dims={"x": 1})
    assert buf.capacity == 10
