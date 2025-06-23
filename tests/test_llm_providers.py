from core import llm_providers
from core.llm_providers.base import LLMProvider


class DummyProvider(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    def generate_response(self, ctx):
        return self.name


def test_manager_selects_provider(tmp_path, monkeypatch):
    cfg = tmp_path / "llm_config.yaml"
    cfg.write_text(
        """default_provider: openai
providers:
  openai:
    type: openai
  local:
    type: local
    model_path: dummy
"""
    )
    monkeypatch.setenv("LLM_CONFIG_PATH", str(cfg))
    monkeypatch.setattr(
        llm_providers.manager, "OpenAIProvider", lambda **k: DummyProvider("openai")
    )
    monkeypatch.setattr(
        llm_providers.manager, "LocalHFProvider", lambda **k: DummyProvider("local")
    )
    mgr = llm_providers.LLMBackendManager()
    assert mgr.get_provider().generate_response(None) == "openai"
    assert mgr.get_provider("local").generate_response(None) == "local"
