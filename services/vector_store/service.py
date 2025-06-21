"""In-memory vector store with embedding via LLM Gateway."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List

import httpx
import numpy as np


class VectorStoreService:
    """Store documents and perform similarity search using embeddings."""

    def __init__(self, llm_url: str = "http://localhost:8003") -> None:
        self.llm_url = llm_url.rstrip("/")
        self.collections: Dict[str, Dict[str, List]] = defaultdict(
            lambda: {"ids": [], "texts": [], "embeddings": []}
        )
        self.provider = "dummy"

    def _embed(self, text: str) -> List[float]:
        try:
            with httpx.Client() as client:
                resp = client.post(f"{self.llm_url}/embed", json={"text": text}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                self.provider = data.get("provider", self.provider)
                return data.get("embedding", [])
        except Exception:  # pragma: no cover - network or service failure
            self.provider = "dummy"
            return [float(len(text))]

    def add_document(self, text: str, collection: str) -> str:
        """Add a document to the given collection."""
        emb = self._embed(text)
        doc_id = str(uuid.uuid4())
        store = self.collections[collection]
        store["ids"].append(doc_id)
        store["texts"].append(text)
        store["embeddings"].append(emb)
        return doc_id

    def search(self, query: str, collection: str, top_k: int = 3) -> Dict[str, Any]:
        """Return the top_k documents similar to the query."""
        q_emb = np.array(self._embed(query))
        store = self.collections.get(collection)
        if not store or not store["embeddings"]:
            return {"matches": [], "model": self.provider}
        embs = np.array(store["embeddings"], dtype=float)
        if embs.ndim == 1:
            embs = embs.reshape(-1, q_emb.shape[0])
        dists = np.linalg.norm(embs - q_emb, axis=1)
        idxs = dists.argsort()[:top_k]
        matches = [
            {
                "id": store["ids"][i],
                "text": store["texts"][i],
                "distance": float(dists[i]),
            }
            for i in idxs
        ]
        return {"matches": matches, "model": self.provider}

