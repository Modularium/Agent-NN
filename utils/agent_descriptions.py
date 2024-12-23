"""Standardized agent descriptions and capabilities."""
from typing import Dict, List, Any

# Detailed descriptions of agent capabilities and domains
AGENT_DESCRIPTIONS = {
    "finance": {
        "name": "Financial Expert Agent",
        "description": """Specialized in financial analysis, planning, and advisory.
        Expertise includes financial modeling, investment analysis, risk assessment,
        and financial reporting.""",
        "capabilities": [
            "Financial analysis and reporting",
            "Investment strategy and portfolio management",
            "Risk assessment and management",
            "Budgeting and forecasting",
            "Tax planning and compliance",
            "Financial modeling and valuation"
        ],
        "knowledge_domains": [
            "Corporate Finance",
            "Investment Banking",
            "Financial Markets",
            "Accounting",
            "Risk Management",
            "Financial Regulations"
        ],
        "example_tasks": [
            "Analyze financial statements and provide insights",
            "Create investment strategies based on risk profiles",
            "Develop financial forecasts and budgets",
            "Calculate key financial metrics and ratios",
            "Assess investment opportunities and risks"
        ]
    },
    "tech": {
        "name": "Technical Expert Agent",
        "description": """Specialized in software development, system architecture,
        and technical problem-solving. Expertise includes programming, system design,
        and technology consulting.""",
        "capabilities": [
            "Software development and architecture",
            "System design and integration",
            "Technical problem-solving",
            "Code review and optimization",
            "Technology stack evaluation",
            "Performance optimization"
        ],
        "knowledge_domains": [
            "Software Engineering",
            "System Architecture",
            "Cloud Computing",
            "DevOps",
            "Database Systems",
            "Security"
        ],
        "example_tasks": [
            "Design system architecture for scalability",
            "Review and optimize code for performance",
            "Implement security best practices",
            "Troubleshoot technical issues",
            "Evaluate and recommend technology solutions"
        ]
    },
    "marketing": {
        "name": "Marketing Expert Agent",
        "description": """Specialized in marketing strategy, digital marketing,
        and brand development. Expertise includes campaign planning, market analysis,
        and customer engagement.""",
        "capabilities": [
            "Marketing strategy development",
            "Digital marketing campaign planning",
            "Brand development and management",
            "Market analysis and research",
            "Customer segmentation",
            "Marketing analytics"
        ],
        "knowledge_domains": [
            "Digital Marketing",
            "Brand Management",
            "Market Research",
            "Customer Behavior",
            "Social Media Marketing",
            "Marketing Analytics"
        ],
        "example_tasks": [
            "Develop comprehensive marketing strategies",
            "Create digital marketing campaigns",
            "Analyze market trends and competition",
            "Design customer engagement programs",
            "Measure and optimize marketing ROI"
        ]
    }
}

def get_agent_description(domain: str) -> Dict[str, Any]:
    """Get the standardized description for an agent domain.
    
    Args:
        domain: The domain to get the description for
        
    Returns:
        Dict containing the agent's description and capabilities
    """
    return AGENT_DESCRIPTIONS.get(domain, {
        "name": f"{domain.title()} Expert Agent",
        "description": f"Specialized in {domain}-related tasks and analysis.",
        "capabilities": [f"General {domain} expertise"],
        "knowledge_domains": [domain.title()],
        "example_tasks": [f"Handle {domain}-related inquiries and tasks"]
    })

def get_agent_embedding_text(domain: str) -> str:
    """Get a concatenated text representation of an agent for embedding.
    
    Args:
        domain: The domain to get the embedding text for
        
    Returns:
        str: Concatenated text describing the agent
    """
    desc = get_agent_description(domain)
    return f"""{desc['name']}. {desc['description']}
    Capabilities: {', '.join(desc['capabilities'])}
    Knowledge Domains: {', '.join(desc['knowledge_domains'])}
    Example Tasks: {', '.join(desc['example_tasks'])}"""

def get_task_requirements(task_description: str) -> List[str]:
    """Extract key requirements from a task description.
    
    Args:
        task_description: Description of the task
        
    Returns:
        List of key requirements identified in the task
    """
    # This is a placeholder for more sophisticated requirement extraction
    # In a real implementation, this could use NLP to identify key requirements
    requirements = []
    
    # Look for common requirement indicators
    if any(word in task_description.lower() for word in ["analyze", "analysis", "evaluate"]):
        requirements.append("analytical_capability")
    if any(word in task_description.lower() for word in ["create", "develop", "design"]):
        requirements.append("creative_capability")
    if any(word in task_description.lower() for word in ["optimize", "improve", "enhance"]):
        requirements.append("optimization_capability")
    if any(word in task_description.lower() for word in ["recommend", "suggest", "advise"]):
        requirements.append("advisory_capability")
        
    return requirements

def match_task_to_domain(task_requirements: List[str], domain: str) -> float:
    """Calculate how well a domain matches task requirements.
    
    Args:
        task_requirements: List of requirements from the task
        domain: Domain to check against
        
    Returns:
        float: Score between 0 and 1 indicating match quality
    """
    domain_desc = get_agent_description(domain)
    capabilities = domain_desc["capabilities"]
    
    # Count how many requirements are met by the domain's capabilities
    matches = sum(1 for req in task_requirements 
                 if any(req.lower() in cap.lower() for cap in capabilities))
    
    # Return score normalized between 0 and 1
    return matches / max(len(task_requirements), 1) if task_requirements else 0.5