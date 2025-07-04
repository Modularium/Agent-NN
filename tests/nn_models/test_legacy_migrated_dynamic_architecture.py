import pytest

pytest.importorskip("torch")


def test_import_dynamic_architecture():
    from nn_models.dynamic_architecture import DynamicArchitecture

    model = DynamicArchitecture(input_dim=6, output_dim=2)
    assert hasattr(model, "forward")
