# LLM Provider Overview

The gateway selects language model providers based on `llm_config.yaml`.
Models can be switched at runtime via the session manager.

Example configuration:
```yaml
default_provider: openai
providers:
  openai:
    type: openai
    api_key: ${OPENAI_API_KEY}
  anthropic:
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
  local:
    type: local
    model_path: ./models/mistral-7b.Q4_K_M.gguf
```
