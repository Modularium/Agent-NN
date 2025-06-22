"""Sample agent worker calling the LLM Gateway."""

from __future__ import annotations

from typing import Any
from datetime import datetime

import httpx

from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT

from core.model_context import ModelContext
from core.audit_log import AuditLog, AuditEntry
from core.crypto import sign_payload


class SampleAgentService:
    """Process tasks using the LLM Gateway and Vector Store."""

    def __init__(
        self,
        llm_url: str = "http://localhost:8003",
        vector_url: str = "http://localhost:8004",
        session_url: str = "http://localhost:8005",
    ) -> None:
        self.llm_url = llm_url.rstrip("/")
        self.vector_url = vector_url.rstrip("/")
        self.session_url = session_url.rstrip("/")
        self.audit = AuditLog()

    def run(self, ctx: ModelContext) -> ModelContext:
        """Invoke the LLM Gateway and return the updated context."""
        start_id = self.audit.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="sample_agent",
                action="task_started",
                context_id=ctx.uuid,
                detail={},
            )
        )
        ctx.audit_trace.append(start_id)
        prompt = ctx.task_context.description or str(ctx.task_context.input_data)
        task_type = ctx.task_context.task_type if ctx.task_context else ""
        if ctx.task_context and ctx.task_context.preferences:
            cid = ctx.task_context.preferences.get("coalition_id")
            title = ctx.task_context.preferences.get("subtask_title")
            if cid:
                prompt = f"[Coalition {cid}] {prompt}"
            if title:
                prompt = f"[{title}] {prompt}"
        documents: list[dict[str, Any]] = []
        semantic = task_type in {"semantic", "qa", "search"}
        if semantic:
            try:
                with httpx.Client() as client:
                    resp = client.post(
                        f"{self.vector_url}/vector_search",
                        json={"query": prompt, "collection": "default", "top_k": 3},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    documents = resp.json().get("matches", [])
            except Exception:
                documents = []
            doc_text = "\n".join(d.get("text", "") for d in documents)
            prompt = f"{prompt}\n\n{doc_text}" if doc_text else prompt

        TOKENS_IN.labels("sample_agent").inc(len(prompt.split()))

        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.llm_url}/generate",
                    json={"prompt": prompt},
                    timeout=10,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except Exception:
            data = {
                "completion": f"Echo: {prompt}",
                "tokens_used": 0,
                "provider": "dummy",
            }
        TOKENS_OUT.labels("sample_agent").inc(data.get("tokens_used", 0))
        TASKS_PROCESSED.labels("sample_agent").inc()

        if semantic:
            avg_dist = (
                sum(d.get("distance", 0.0) for d in documents) / len(documents)
                if documents
                else 0.0
            )
            ctx.result = {
                "generated_response": data["completion"],
                "sources": documents,
                "embedding_distance_avg": avg_dist,
            }
        else:
            ctx.result = data["completion"]
        ctx.metrics = {"tokens_used": data.get("tokens_used", 0)}
        if ctx.session_id:
            try:
                with httpx.Client() as client:
                    client.post(
                        f"{self.session_url}/update_context",
                        json=ctx.model_dump(),
                        timeout=5,
                    )
            except Exception:
                pass
        end_id = self.audit.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="sample_agent",
                action="task_completed",
                context_id=ctx.uuid,
                detail={"tokens": data.get("tokens_used", 0)},
            )
        )
        ctx.audit_trace.append(end_id)
        sig = sign_payload(
            "sample_agent", ctx.model_dump(exclude={"signature", "signed_by"})
        )
        ctx.signed_by = sig["signed_by"]
        ctx.signature = sig["signature"]
        return ctx
