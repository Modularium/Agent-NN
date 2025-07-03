# Prompt Refinement

The `prompt_refiner` module provides simple utilities to optimise prompts before they are sent to an agent or language model.

## Strategies

- **Direct Rewrite** – normalises whitespace and removes superfluous characters.
- **Shorten** – trims the prompt to a maximum length of 200 characters.

## Usage

```python
from agentnn.prompting import propose_refinement, evaluate_prompt_quality

refined = propose_refinement("   translate  this   text  ")
score = evaluate_prompt_quality(refined)
```
