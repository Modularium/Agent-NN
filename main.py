from agents.chatbot_agent import ChatbotAgent
from agents.supervisor_agent import SupervisorAgent

def main():
    supervisor = SupervisorAgent()
    chatbot = ChatbotAgent(supervisor)

    user_message = "Hallo, bitte finde alle Kunden mit offenen Rechnungen."
    response = chatbot.handle_user_message(user_message)
    print("Chatbot Antwort:", response)

if __name__ == "__main__":
    main()
