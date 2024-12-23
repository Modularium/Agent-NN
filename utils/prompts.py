from langchain.prompts import PromptTemplate

# Chatbot system prompt for general conversation
CHATBOT_SYSTEM_PROMPT = PromptTemplate(
    template="""You are a friendly and helpful AI assistant. You engage in natural conversation while also being able to help with specific tasks.
    Always maintain a professional and courteous tone. If the user's request seems to require specialized knowledge or complex task execution,
    recommend delegating it to a specialized agent.

    Current conversation:
    {chat_history}
    
    User: {input}
    Assistant:""",
    input_variables=["input", "chat_history"]
)

# Prompt for task identification
TASK_IDENTIFICATION_PROMPT = PromptTemplate(
    template="""Analyze the following user message and determine if it contains a task request that should be delegated to a specialized agent.
    A task request typically involves:
    1. A specific action or goal to accomplish
    2. Requirements or constraints
    3. Expected output or deliverables

    User message: {message}

    Is this a task request? Respond with either:
    TASK: [task description] if it is a task
    CHAT: [reason] if it's just conversation

    Response:""",
    input_variables=["message"]
)

# Prompt for task result formatting
TASK_RESULT_PROMPT = PromptTemplate(
    template="""Format the following task execution result in a user-friendly way.
    Add relevant context and explanations where needed.

    Task: {task_description}
    Raw Result: {result}

    Format the response to be clear, informative, and easy to understand.
    If there were any errors, explain them in user-friendly terms.

    Formatted Response:""",
    input_variables=["task_description", "result"]
)

# Prompt for error handling
ERROR_HANDLING_PROMPT = PromptTemplate(
    template="""An error occurred while processing your request. I'll explain what happened and suggest next steps.

    Original Request: {user_request}
    Error: {error_message}

    Let me help you understand what happened and how we can proceed:""",
    input_variables=["user_request", "error_message"]
)
