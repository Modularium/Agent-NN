import pytest

pytest.importorskip("torch")


def test_import_advanced_training():
    from nn_models.advanced_training import HierarchicalNetwork

    net = HierarchicalNetwork(text_dim=2, metric_dim=2, feedback_dim=2, hidden_dims=[2,2,2,2])
    assert hasattr(net, 'forward')
