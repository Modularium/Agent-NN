import importlib.util
import sys
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "context_reasoner",
    str(Path(__file__).resolve().parents[2] / "agentnn" / "reasoning" / "context_reasoner.py"),
)
module = importlib.util.module_from_spec(spec)
sys.modules["context_reasoner"] = module
assert spec.loader
spec.loader.exec_module(module)
MajorityVoteReasoner = module.MajorityVoteReasoner


@pytest.mark.unit
def test_majority_vote():
    reasoner = MajorityVoteReasoner()
    reasoner.add_step("a1", "x", 0.6)
    reasoner.add_step("a2", "y", 0.2)
    reasoner.add_step("a3", "x", 0.9)
    assert reasoner.decide() == "x"
