import json
import urllib.request
from types import SimpleNamespace, ModuleType
import sys

dummy = ModuleType("langchain_openai")
dummy.OpenAI = object
dummy.ChatOpenAI = object
dummy.OpenAIEmbeddings = object
sys.modules.setdefault("langchain_openai", dummy)
sys.modules.setdefault("requests", ModuleType("requests"))
backend_stub = ModuleType("llm_models.llm_backend")
backend_stub.LLMBackendManager = lambda: SimpleNamespace(get_llm=lambda: SimpleNamespace(invoke=lambda p: "ok"))
sys.modules.setdefault("llm_models.llm_backend", backend_stub)
from mcp.llm_gateway.service import LLMGatewayService


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def fake_urlopen(req, timeout=0):
    return FakeResponse([])


def test_generate(monkeypatch):
    service = LLMGatewayService()
    fake_llm = SimpleNamespace(invoke=lambda prompt: "resp:" + prompt)
    monkeypatch.setattr(service, "backend", SimpleNamespace(get_llm=lambda: fake_llm))
    text = service.generate("hi")
    assert text.startswith("resp:")


def test_qa_timeout(monkeypatch):
    service = LLMGatewayService()
    fake_llm = SimpleNamespace(invoke=lambda prompt: "ans")
    monkeypatch.setattr(service, "backend", SimpleNamespace(get_llm=lambda: fake_llm))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    text = service.qa("question")
    assert text == "ans"


def test_translate(monkeypatch):
    service = LLMGatewayService()
    fake_llm = SimpleNamespace(invoke=lambda prompt: "translated")
    monkeypatch.setattr(service, "backend", SimpleNamespace(get_llm=lambda: fake_llm))
    text = service.translate("hi", "de")
    assert text == "translated"


def test_vision_describe(monkeypatch):
    service = LLMGatewayService()
    fake_llm = SimpleNamespace(invoke=lambda prompt: "description")
    monkeypatch.setattr(service, "backend", SimpleNamespace(get_llm=lambda: fake_llm))
    text = service.vision_describe("http://img")
    assert text == "description"
