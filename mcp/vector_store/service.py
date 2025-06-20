"""In-memory vector store using HuggingFace embeddings."""

from __future__ import annotations

import uuid
from typing import Dict, List

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreService:
    """Store documents and perform similarity search."""

    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}
        self._embeddings: List[List[float]] = []
        self._ids: List[str] = []
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    def add_document(self, document: Dict) -> Dict:
        """Add a text document to the store."""
        doc_id = document.get("id") or str(uuid.uuid4())
        text = document.get("text", "")
        embedding = self.embedder.embed_query(text)
        self._docs[doc_id] = text
        self._ids.append(doc_id)
        self._embeddings.append(embedding)
        return {"id": doc_id}

    def query(self, query: str) -> List[Dict]:
        """Return documents most similar to the query."""
        if not self._embeddings:
            return []
        q_emb = np.array(self.embedder.embed_query(query))
        embs = np.array(self._embeddings)
        sims = embs @ q_emb
        idxs = sims.argsort()[-3:][::-1]
        results = []
        for i in idxs:
            doc_id = self._ids[i]
            results.append(
                {"id": doc_id, "text": self._docs[doc_id], "score": float(sims[i])}
            )
        return results
