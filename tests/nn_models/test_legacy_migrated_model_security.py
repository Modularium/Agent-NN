import pytest

pytest.importorskip("torch")


def test_import_model_security():
    from nn_models.model_security import ModelSecurityManager

    sec = ModelSecurityManager()
    assert hasattr(sec, 'validate_checksum')
