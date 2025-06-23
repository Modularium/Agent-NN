from core.llm_providers.base import LLMProvider
from core.model_context import ModelContext
from services.llm_gateway.service import LLMGatewayService


class DummyProvider(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    def generate_response(self, ctx: ModelContext) -> str:
        return self.name


def test_switching(monkeypatch):
    monkeypatch.setenv("LLM_CONFIG_PATH", "nonexistent")
    gateway = LLMGatewayService()
    monkeypatch.setattr(
        gateway.manager,
        "get_provider",
        lambda name=None: DummyProvider(name or "openai"),
    )
    gateway.session_mgr.set_model("u1", "local")
    ctx = ModelContext(task="hi", user_id="u1")
    assert gateway.chat(ctx)["provider"] == "local"
    gateway.session_mgr.set_model("u1", "openai")
    assert gateway.chat(ctx)["provider"] == "openai"
