from .base_llm import BaseLLM

class SpecializedLLM:
    def __init__(self, domain: str):
        # Hier könnte man z.B. ein Fine-Tuning Modell laden
        # Oder differenzierte Prompt-Templates nutzen
        self.domain = domain
        # Für Demo: einfaches Base LLM
        self.base_llm = BaseLLM(temperature=0.2)

    def get_llm(self):
        # Könnte in Zukunft z.B. nach Domain-Model differenzieren
        return self.base_llm.get_llm()
