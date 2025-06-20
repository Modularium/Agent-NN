from typing import List, Dict


class VectorStoreService:
    """Stub vector store."""

    def query(self, query: str) -> List[Dict]:
        return []

    def add_document(self, document: Dict) -> Dict:
        return {"id": "doc1"}
