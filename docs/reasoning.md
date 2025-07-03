# Context Reasoning

The reasoning framework collects partial results from multiple agents and derives
an aggregated decision. Steps are registered via `add_step()` and finally
evaluated by calling `decide()` on the selected strategy.

## Strategies

- **MajorityVoteReasoner** – sums up the scores of identical results and returns
the most popular answer.

```python
from agentnn.reasoning import MajorityVoteReasoner

reasoner = MajorityVoteReasoner()
reasoner.add_step("a1", "yes", 0.8)
reasoner.add_step("a2", "no", 0.5)
reasoner.add_step("a3", "yes")
assert reasoner.decide() == "yes"
```
