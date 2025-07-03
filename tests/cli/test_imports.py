import pytest
import sys
import types


@pytest.mark.unit
def test_no_circular_imports():
    sys.modules.setdefault(
        "mlflow",
        types.SimpleNamespace(
            start_run=lambda *a, **k: None,
            set_tag=lambda *a, **k: None,
            log_param=lambda *a, **k: None,
            log_metric=lambda *a, **k: None,
            set_tracking_uri=lambda *a, **k: None,
            tracking=types.SimpleNamespace(
                MlflowClient=lambda: types.SimpleNamespace(
                    list_experiments=lambda: [],
                    get_run=lambda run_id: types.SimpleNamespace(
                        info=types.SimpleNamespace(run_id=run_id, status="FINISHED"),
                        data=types.SimpleNamespace(metrics={}, params={}),
                    ),
                )
            ),
        ),
    )
    dummy_mcp = types.ModuleType("mcp")
    class DummyClient:
        def __init__(self, *a, **k) -> None:
            pass

    dummy_mcp.MCPClient = DummyClient
    dummy_mcp.create_app = lambda *a, **k: None
    dummy_mcp.to_mcp = lambda x: x
    dummy_mcp.from_mcp = lambda x: x
    dummy_mcp_ws = types.ModuleType("mcp_ws")
    dummy_mcp_ws.ws_server = types.SimpleNamespace(broadcast=lambda *a, **k: None)
    sys.modules.setdefault("agentnn.mcp", dummy_mcp)
    sys.modules.setdefault("agentnn.mcp.mcp_client", types.SimpleNamespace(MCPClient=DummyClient))
    sys.modules.setdefault("agentnn.mcp.mcp_server", types.SimpleNamespace(create_app=lambda: None))
    sys.modules.setdefault("agentnn.mcp.mcp_ws", dummy_mcp_ws)
    import sdk.nn_models as _nn
    sys.modules.setdefault("sdk.cli.nn_models", _nn)
    DummySettings = type(
        "DummySettings",
        (),
        {"load": classmethod(lambda cls: cls()), "__init__": lambda self: None},
    )
    sys.modules.setdefault("sdk.cli.config", types.SimpleNamespace(SDKSettings=DummySettings))
    sys.modules.setdefault(
        "core.config", types.SimpleNamespace(settings=types.SimpleNamespace(model_dump=lambda: {}))
    )
    import sdk.cli.commands.session
    import sdk.cli.commands.agent
    import sdk.cli.main

    assert sdk.cli.commands.session is not None
    assert sdk.cli.commands.agent is not None
    assert sdk.cli.main is not None
