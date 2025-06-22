"""In-memory vector store with embedding via LLM Gateway."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List

import httpx
import numpy as np
import chromadb

from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT
from core.config import settings


class VectorStoreService:
    """Store documents and perform similarity search using embeddings."""

    def __init__(self, llm_url: str = "http://localhost:8003") -> None:
        self.llm_url = llm_url.rstrip("/")
        self.provider = "dummy"
        backend = settings.VECTOR_DB_BACKEND.lower()
        if backend == "chromadb":
            self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
            self.collections = {}
        else:
            self.client = None
            self.collections: Dict[str, Dict[str, List]] = defaultdict(
                lambda: {"ids": [], "texts": [], "embeddings": []}
            )

    def _embed(self, text: str) -> List[float]:
        try:
            with httpx.Client() as client:
                resp = client.post(f"{self.llm_url}/embed", json={"text": text}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                self.provider = data.get("provider", self.provider)
                emb = data.get("embedding", [])
                TOKENS_IN.labels("vector_store").inc(len(text.split()))
                TOKENS_OUT.labels("vector_store").inc(len(emb))
                return emb
        except Exception:  # pragma: no cover - network or service failure
            self.provider = "dummy"
            TOKENS_IN.labels("vector_store").inc(len(text.split()))
            TOKENS_OUT.labels("vector_store").inc(1)
            return [float(len(text))]

    def embed(self, text: str) -> Dict[str, Any]:
        """Return embedding and provider for given text."""
        emb = self._embed(text)
        return {"embedding": emb, "model": self.provider}

    def add_document(self, text: str, collection: str) -> str:
        """Add a document to the given collection."""
        emb = self._embed(text)
        doc_id = str(uuid.uuid4())
        if self.client:
            coll = self.client.get_or_create_collection(collection)
            coll.add(ids=[doc_id], documents=[text], embeddings=[emb])
        else:
            store = self.collections[collection]
            store["ids"].append(doc_id)
            store["texts"].append(text)
            store["embeddings"].append(emb)
        TASKS_PROCESSED.labels("vector_store").inc()
        return doc_id

    def search(self, query: str, collection: str, top_k: int = 3) -> Dict[str, Any]:
        """Return the top_k documents similar to the query."""
        q_emb = np.array(self._embed(query))
        if self.client:
            coll = self.client.get_or_create_collection(collection)
            res = coll.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
            matches = [
                {
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "distance": float(res["distances"][0][i]),
                }
                for i in range(len(res["ids"][0]))
            ]
        else:
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
        TASKS_PROCESSED.labels("vector_store").inc()
        return {"matches": matches, "model": self.provider}

