import importlib.util
import sys
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "tool_vote", str(Path(__file__).resolve().parents[2] / "agentnn" / "reasoning" / "tool_vote.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules["tool_vote"] = module
assert spec.loader
spec.loader.exec_module(module)  # type: ignore
ToolResultVote = module.ToolResultVote


@pytest.mark.unit
def test_tool_vote_decide():
    vote = ToolResultVote()
    vote.add_result("t1", "a", confidence=0.5, relevance=0.5)
    vote.add_result("t2", "b", confidence=0.9, relevance=0.9)
    vote.add_result("t1", "c", confidence=0.6, relevance=0.6)
    best = vote.decide()
    assert best is not None
    assert best.tool == "t1" or best.tool == "t2"
    assert best.output in {"a", "b", "c"}
