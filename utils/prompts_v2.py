"""Prompt templates using latest LangChain components."""
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Domain-specific system messages
DOMAIN_SYSTEM_MESSAGES = {
    "finance": SystemMessage(content="""You are a financial expert assistant with deep knowledge in:
- Financial analysis and reporting
- Investment strategies
- Risk assessment
- Tax planning
- Budgeting and forecasting

Use precise financial terminology and always consider:
1. Data accuracy and sources
2. Risk factors and disclaimers
3. Regulatory compliance
4. Time sensitivity of information"""),

    "tech": SystemMessage(content="""You are a technical expert assistant specializing in:
- Software development
- System architecture
- Cloud infrastructure
- DevOps practices
- Security considerations

Follow these guidelines:
1. Provide code examples when relevant
2. Consider security implications
3. Follow best practices
4. Include error handling
5. Consider scalability"""),

    "marketing": SystemMessage(content="""You are a marketing strategy expert with expertise in:
- Market analysis
- Brand development
- Campaign planning
- Customer engagement
- Digital marketing

Focus on:
1. Target audience understanding
2. Brand consistency
3. ROI metrics
4. Channel optimization
5. Engagement strategies""")
}

# Task-specific templates
TASK_TEMPLATES = {
    "finance": {
        "investment_analysis": ChatPromptTemplate.from_messages([
            DOMAIN_SYSTEM_MESSAGES["finance"],
            HumanMessage(content="""Analyze this investment opportunity:

Context: {context}

Consider:
1. Risk factors
2. Expected returns
3. Market conditions
4. Investment timeline
5. Portfolio fit

Question: {question}""")
        ]),
        
        "financial_report": ChatPromptTemplate.from_messages([
            DOMAIN_SYSTEM_MESSAGES["finance"],
            HumanMessage(content="""Review this financial data:

Context: {context}

Analyze:
1. Key performance indicators
2. Trend analysis
3. Comparative metrics
4. Risk factors
5. Future projections

Question: {question}""")
        ])
    },
    
    "tech": {
        "code_review": ChatPromptTemplate.from_messages([
            DOMAIN_SYSTEM_MESSAGES["tech"],
            HumanMessage(content="""Review this code:

Context: {context}

Check for:
1. Code quality
2. Security issues
3. Performance optimizations
4. Best practices
5. Documentation needs

Question: {question}""")
        ]),
        
        "architecture_design": ChatPromptTemplate.from_messages([
            DOMAIN_SYSTEM_MESSAGES["tech"],
            HumanMessage(content="""Analyze this system architecture:

Context: {context}

Consider:
1. Scalability
2. Security
3. Performance
4. Maintainability
5. Cost efficiency

Question: {question}""")
        ])
    },
    
    "marketing": {
        "campaign_analysis": ChatPromptTemplate.from_messages([
            DOMAIN_SYSTEM_MESSAGES["marketing"],
            HumanMessage(content="""Analyze this marketing campaign:

Context: {context}

Evaluate:
1. Target audience reach
2. Channel performance
3. Engagement metrics
4. ROI analysis
5. Improvement opportunities

Question: {question}""")
        ]),
        
        "market_research": ChatPromptTemplate.from_messages([
            DOMAIN_SYSTEM_MESSAGES["marketing"],
            HumanMessage(content="""Review this market data:

Context: {context}

Analyze:
1. Market trends
2. Competitor analysis
3. Customer segments
4. Growth opportunities
5. Risk factors

Question: {question}""")
        ])
    }
}

# Core chat templates
CHATBOT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a friendly and helpful AI assistant. You engage in natural conversation while also being able to help with specific tasks.
Always maintain a professional and courteous tone. If the user's request seems to require specialized knowledge or complex task execution,
recommend delegating it to a specialized agent."""),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}")
])

TASK_IDENTIFICATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content="""Analyze user messages to identify task requests that should be delegated to specialized agents.
A task request typically involves:
1. A specific action or goal to accomplish
2. Requirements or constraints
3. Expected output or deliverables

Respond with either:
TASK: [task description] if it is a task
CHAT: [reason] if it's just conversation"""),
    HumanMessage(content="User message: {message}")
])

TASK_RESULT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content="""Format task execution results in a user-friendly way.
Add relevant context and explanations where needed.
Make the response clear, informative, and easy to understand.
If there were any errors, explain them in user-friendly terms."""),
    HumanMessage(content="""Task: {task_description}
Raw Result: {result}""")
])

ERROR_HANDLING_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content="Explain errors in user-friendly terms and suggest next steps."),
    HumanMessage(content="""An error occurred while processing your request.

Original Request: {user_request}
Error: {error_message}""")
])

def format_chat_history(history: List[Dict[str, str]]) -> List[Any]:
    """Format chat history into a list of messages.
    
    Args:
        history: List of chat messages
        
    Returns:
        List of formatted messages
    """
    formatted = []
    for msg in history:
        if msg["role"] == "user":
            formatted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted.append(AIMessage(content=msg["content"]))
    return formatted

def get_domain_template(domain: str) -> ChatPromptTemplate:
    """Get base template for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        ChatPromptTemplate for the domain
    """
    if domain not in DOMAIN_SYSTEM_MESSAGES:
        raise ValueError(f"Unknown domain: {domain}")
        
    return ChatPromptTemplate.from_messages([
        DOMAIN_SYSTEM_MESSAGES[domain],
        HumanMessage(content="""Context: {context}
Question: {question}""")
    ])

def get_task_template(domain: str, task_type: str) -> ChatPromptTemplate:
    """Get specialized template for a task.
    
    Args:
        domain: Domain name
        task_type: Type of task
        
    Returns:
        ChatPromptTemplate for the task
    """
    if domain not in TASK_TEMPLATES:
        raise ValueError(f"Unknown domain: {domain}")
        
    if task_type not in TASK_TEMPLATES[domain]:
        raise ValueError(f"Unknown task type: {task_type}")
        
    return TASK_TEMPLATES[domain][task_type]

def create_combined_template(domain: str,
                           task_type: Optional[str] = None,
                           additional_context: Optional[Dict[str, Any]] = None) -> ChatPromptTemplate:
    """Create a combined template with domain and task specifics.
    
    Args:
        domain: Domain name
        task_type: Optional task type
        additional_context: Optional additional context
        
    Returns:
        ChatPromptTemplate: Combined template
    """
    messages = [DOMAIN_SYSTEM_MESSAGES[domain]]
    
    # Add task-specific system message if provided
    if task_type and domain in TASK_TEMPLATES:
        if task_type in TASK_TEMPLATES[domain]:
            task_template = TASK_TEMPLATES[domain][task_type]
            messages.extend(task_template.messages[1:])  # Skip system message
            
    # Add additional context if provided
    context_parts = ["Context: {context}", "Question: {question}"]
    if additional_context:
        context_str = "\n".join(f"{k}: {v}" for k, v in additional_context.items())
        context_parts.insert(1, f"Additional Context:\n{context_str}")
        
    messages.append(HumanMessage(content="\n\n".join(context_parts)))
    
    return ChatPromptTemplate.from_messages(messages)