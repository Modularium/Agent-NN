"""Prompt templates for different components of the system."""
from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate

# Domain-specific base templates
DOMAIN_TEMPLATES = {
    "finance": """You are a financial expert assistant with deep knowledge in:
- Financial analysis and reporting
- Investment strategies
- Risk assessment
- Tax planning
- Budgeting and forecasting

Use precise financial terminology and always consider:
1. Data accuracy and sources
2. Risk factors and disclaimers
3. Regulatory compliance
4. Time sensitivity of information

Context: {context}

Question: {question}

Provide a well-structured response with clear recommendations and necessary disclaimers.""",

    "tech": """You are a technical expert assistant specializing in:
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
5. Consider scalability

Context: {context}

Question: {question}

Provide a detailed technical response with practical examples and considerations.""",

    "marketing": """You are a marketing strategy expert with expertise in:
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
5. Engagement strategies

Context: {context}

Question: {question}

Provide strategic marketing advice with actionable insights and metrics.""",

    "legal": """You are a legal expert assistant specializing in:
- Contract law
- Regulatory compliance
- Corporate law
- Intellectual property
- Risk management

Always consider:
1. Jurisdiction specifics
2. Recent legal changes
3. Precedent cases
4. Compliance requirements
5. Risk factors

Context: {context}

Question: {question}

Provide legal guidance while noting this is not formal legal advice."""
}

# Specialized task templates
TASK_TEMPLATES = {
    "finance": {
        "investment_analysis": """Analyze the following investment opportunity:

Context: {context}

Consider:
1. Risk factors
2. Expected returns
3. Market conditions
4. Investment timeline
5. Portfolio fit

Question: {question}

Provide a structured analysis with clear recommendations.""",

        "financial_report": """Review the following financial data:

Context: {context}

Analyze:
1. Key performance indicators
2. Trend analysis
3. Comparative metrics
4. Risk factors
5. Future projections

Question: {question}

Provide a comprehensive financial report with insights."""
    },
    
    "tech": {
        "code_review": """Review the following code:

Context: {context}

Check for:
1. Code quality
2. Security issues
3. Performance optimizations
4. Best practices
5. Documentation needs

Question: {question}

Provide detailed code review feedback with examples.""",

        "architecture_design": """Analyze the following system architecture:

Context: {context}

Consider:
1. Scalability
2. Security
3. Performance
4. Maintainability
5. Cost efficiency

Question: {question}

Provide architectural recommendations with diagrams if needed."""
    },
    
    "marketing": {
        "campaign_analysis": """Analyze the following marketing campaign:

Context: {context}

Evaluate:
1. Target audience reach
2. Channel performance
3. Engagement metrics
4. ROI analysis
5. Improvement opportunities

Question: {question}

Provide campaign insights and optimization recommendations.""",

        "market_research": """Review the following market data:

Context: {context}

Analyze:
1. Market trends
2. Competitor analysis
3. Customer segments
4. Growth opportunities
5. Risk factors

Question: {question}

Provide market insights with actionable recommendations."""
    }
}

def get_domain_template(domain: str) -> PromptTemplate:
    """Get base template for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        PromptTemplate for the domain
    """
    if domain not in DOMAIN_TEMPLATES:
        raise ValueError(f"Unknown domain: {domain}")
        
    return PromptTemplate(
        template=DOMAIN_TEMPLATES[domain],
        input_variables=["context", "question"]
    )

def get_task_template(domain: str, task_type: str) -> PromptTemplate:
    """Get specialized template for a task.
    
    Args:
        domain: Domain name
        task_type: Type of task
        
    Returns:
        PromptTemplate for the task
    """
    if domain not in TASK_TEMPLATES:
        raise ValueError(f"Unknown domain: {domain}")
        
    if task_type not in TASK_TEMPLATES[domain]:
        raise ValueError(f"Unknown task type: {task_type}")
        
    return PromptTemplate(
        template=TASK_TEMPLATES[domain][task_type],
        input_variables=["context", "question"]
    )

def create_combined_prompt(domain: str,
                         task_type: Optional[str] = None,
                         additional_context: Optional[Dict[str, Any]] = None) -> PromptTemplate:
    """Create a combined prompt with domain and task specifics.
    
    Args:
        domain: Domain name
        task_type: Optional task type
        additional_context: Optional additional context
        
    Returns:
        PromptTemplate: Combined prompt template
    """
    # Get base template
    base_template = DOMAIN_TEMPLATES[domain]
    
    # Add task-specific template if provided
    if task_type and domain in TASK_TEMPLATES:
        if task_type in TASK_TEMPLATES[domain]:
            task_template = TASK_TEMPLATES[domain][task_type]
            base_template = f"{base_template}\n\nTask-Specific Instructions:\n{task_template}"
            
    # Add additional context if provided
    if additional_context:
        context_str = "\n".join(f"{k}: {v}" for k, v in additional_context.items())
        base_template = f"{base_template}\n\nAdditional Context:\n{context_str}"
        
    return PromptTemplate(
        template=base_template,
        input_variables=["context", "question"]
    )

def get_system_prompt(domain: str) -> str:
    """Get system prompt for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        str: System prompt
    """
    if domain not in DOMAIN_TEMPLATES:
        raise ValueError(f"Unknown domain: {domain}")
        
    # Extract the role description (first paragraph)
    lines = DOMAIN_TEMPLATES[domain].split("\n\n")
    return lines[0]

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
