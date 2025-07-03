import importlib.util
import sys
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "prompt_refiner", str(Path(__file__).resolve().parents[2] / "agentnn" / "prompting" / "prompt_refiner.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules["prompt_refiner"] = module
assert spec.loader
spec.loader.exec_module(module)  # type: ignore
propose_refinement = module.propose_refinement
evaluate_prompt_quality = module.evaluate_prompt_quality


@pytest.mark.unit
def test_propose_refinement():
    assert propose_refinement("  hi  there  ") == "hi there"
    assert propose_refinement("test", strategy="shorten") == "test"


@pytest.mark.unit
def test_evaluate_prompt_quality():
    assert evaluate_prompt_quality("") == 0.0
    assert evaluate_prompt_quality("short") == 0.3
    assert evaluate_prompt_quality("this is a somewhat longer prompt") >= 0.7
