"""Utilities for API routes."""
from functools import wraps
from typing import Callable, Any

def api_route(version: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to mark API routes with a version."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func.api_version = version
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator
