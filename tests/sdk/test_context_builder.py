import sys
from types import ModuleType, SimpleNamespace

stub = ModuleType('core.model_context')
stub.ModelContext = SimpleNamespace
stub.TaskContext = SimpleNamespace
sys.modules['core.model_context'] = stub

from sdk.utils import build_context


def test_build_context():
    ctx = build_context("do something", session_id="abc")
    assert ctx.session_id == "abc"
    assert ctx.task == "do something"
