from enum import Enum


class AgentRole(str, Enum):
    """Predefined roles for agent authorization."""

    WRITER = "writer"
    RETRIEVER = "retriever"
    CRITIC = "critic"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    COACH = "coach"


ROLE_HIERARCHY: dict[str, list[str]] = {
    "reviewer": ["reader"],
    "critic": ["reviewer"],
    "analyst": ["critic", "reader"],
    "coordinator": ["analyst", "writer", "retriever"],
    "coach": ["reviewer"],
}


def _expand_roles(roles: list[str]) -> set[str]:
    result = set(roles)
    stack = list(roles)
    while stack:
        role = stack.pop()
        for parent in ROLE_HIERARCHY.get(role, []):
            if parent not in result:
                result.add(parent)
                stack.append(parent)
    return result


def resolve_roles(agent: str) -> set[str]:
    from .governance import AgentContract

    contract = AgentContract.load(agent)
    roles = contract.allowed_roles + (contract.temp_roles or [])
    return _expand_roles(roles)
