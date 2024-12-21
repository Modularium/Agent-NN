from langchain import PromptTemplate

CHATBOT_SYSTEM_PROMPT = PromptTemplate.from_template(
    "Du bist ein freundlicher Chatbot. Der Nutzer sagt: {input}"
)
