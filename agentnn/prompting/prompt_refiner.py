"""Utility functions to refine and evaluate prompts."""

from __future__ import annotations

__all__ = ["propose_refinement", "evaluate_prompt_quality"]


def propose_refinement(prompt: str, strategy: str = "direct") -> str:
    """Return an improved version of the given prompt.

    Parameters
    ----------
    prompt:
        Original prompt string.
    strategy:
        Refinement strategy. ``direct`` simply normalises whitespace.
        ``shorten`` truncates to 200 characters. ``noop`` returns the input unchanged.
    """
    if strategy == "direct":
        return " ".join(prompt.split())
    if strategy == "shorten":
        return prompt.strip()[:200]
    return prompt


def evaluate_prompt_quality(prompt: str) -> float:
    """Return a simple quality score between 0 and 1."""
    length = len(prompt.strip())
    if length == 0:
        return 0.0
    if length < 20:
        return 0.3
    if length < 100:
        return 0.7
    return 1.0
