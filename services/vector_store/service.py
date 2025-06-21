"""Vector search service stub."""

from typing import Any, Dict


class VectorStoreService:
    """Provide rudimentary vector search functionality."""

    def search(self, text: str, embedding_model: str | None = None) -> Dict[str, Any]:
        """Return a dummy search result."""
        return {"matches": [], "model": embedding_model or "default"}
