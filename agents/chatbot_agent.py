from langchain.chains import LLMChain
from langchain.llms import OpenAI
from utils.prompts import CHATBOT_SYSTEM_PROMPT
from config import LLM_API_KEY

class ChatbotAgent:
    def __init__(self, supervisor_agent):
        # Einfaches LLM für Interaktion mit dem Nutzer
        self.llm = OpenAI(openai_api_key=LLM_API_KEY, temperature=0.7)
        self.chain = LLMChain(llm=self.llm, prompt=CHATBOT_SYSTEM_PROMPT)
        self.supervisor = supervisor_agent

    def handle_user_message(self, user_message: str):
        # Chatbot entscheidet: Ist es nur ein Chat oder eine Aufgabe?
        # Einfacher Heuristik: Wenn "Bitte" und "aufgabe" vorkommt, delegieren
        if "bitte" in user_message.lower() or "aufgabe" in user_message.lower():
            # Extrahiere Task (vereinfachter Dummy)
            task_description = user_message
            result = self.supervisor.execute_task(task_description)
            # Ergebnis an den Nutzer zurückgeben
            return f"Ergebnis: {result}"
        else:
            # Normale Konversation
            response = self.chain.run(user_message)
            return response
