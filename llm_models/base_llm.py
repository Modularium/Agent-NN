from langchain.llms import OpenAI
from config import LLM_API_KEY

class BaseLLM:
    def __init__(self, temperature=0.0):
        self.llm = OpenAI(openai_api_key=LLM_API_KEY, temperature=temperature)

    def get_llm(self):
        return self.llm
