from services.llm_gateway.service import LLMGatewayService


def test_embed_returns_vector():
    service = LLMGatewayService()
    res = service.embed("hello")
    assert isinstance(res["embedding"], list)
    assert res["provider"] in {"local", "dummy"}
