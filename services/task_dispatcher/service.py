"""Task dispatcher core logic."""

import logging
import os
import time
from datetime import datetime
from typing import Any, List

import httpx

from core.access_control import is_authorized
from core.audit_log import AuditEntry, AuditLog
from core.crypto import verify_signature
from core.dispatch_queue import DispatchQueue
from core.governance import AgentContract
from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT
from core.model_context import AgentRunContext, ModelContext, TaskContext
from core.privacy_filter import filter_permissions, redact_context
from core.roles import resolve_roles
from core.trust_evaluator import calculate_trust, update_trust_usage
from core.agent_profile import AgentIdentity
from core.skill_matcher import match_agent_to_task
from core.role_capabilities import apply_role_capabilities

from .config import settings


class TaskDispatcherService:
    """Dispatch incoming tasks to worker agents."""

    def __init__(
        self,
        registry_url: str | None = None,
        session_url: str | None = None,
        coordinator_url: str | None = None,
        coalition_url: str | None = None,
    ) -> None:
        self.registry_url = (registry_url or settings.registry_url).rstrip("/")
        self.session_url = (session_url or settings.session_url).rstrip("/")
        self.coordinator_url = (coordinator_url or settings.coordinator_url).rstrip("/")
        self.coalition_url = (coalition_url or settings.coalition_url).rstrip("/")
        self.queue = DispatchQueue()
        self.log = logging.getLogger(__name__)
        self.audit = AuditLog()

    def _apply_role_limits(self, ctx: ModelContext, role: str) -> None:
        """Limit context according to ROLE_CAPABILITIES."""
        before_tokens = ctx.max_tokens
        before_mem = len(ctx.memory or [])
        apply_role_capabilities(ctx, role)
        if ctx.max_tokens != before_tokens or len(ctx.memory or []) != before_mem:
            ctx.warning = ctx.warning or "role_limits_applied"
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="role_limits_applied",
                    context_id=ctx.uuid,
                    detail={"role": role},
                )
            )
            ctx.audit_trace.append(log_id)

    def _governance_allowed(self, agent: dict[str, Any], ctx: ModelContext) -> bool:
        contract = AgentContract.load(agent["name"])
        history = (
            ctx.task_context.preferences.get("history", [])
            if ctx.task_context and ctx.task_context.preferences
            else []
        )
        trust = calculate_trust(agent["name"], history)
        if trust < contract.trust_level_required:
            ctx.warning = "trust level too low"
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="trust_rejected",
                    context_id=ctx.uuid,
                    detail={"agent": agent["name"], "trust": trust},
                )
            )
            ctx.audit_trace.append(log_id)
            return False
        allowed_roles = resolve_roles(agent["name"])
        if allowed_roles and agent.get("role") not in allowed_roles:
            ctx.warning = "role not allowed"
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="role_rejected",
                    context_id=ctx.uuid,
                    detail={"agent": agent["name"], "role": agent.get("role")},
                )
            )
            ctx.audit_trace.append(log_id)
            return False
        if ctx.mission_role and agent.get("role") != ctx.mission_role:
            ctx.warning = "mission_role_mismatch"
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="mission_role_mismatch",
                    context_id=ctx.uuid,
                    detail={"required": ctx.mission_role, "agent": agent["name"]},
                )
            )
            ctx.audit_trace.append(log_id)
            return False
        if contract.temp_roles and agent.get("role") in contract.temp_roles:
            ctx.elevated_roles.append(agent.get("role"))
            contract.temp_roles.remove(agent.get("role"))
            contract.save()
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="temp_role_used",
                    context_id=ctx.uuid,
                    detail={"agent": agent["name"], "role": agent.get("role")},
                )
            )
            ctx.audit_trace.append(log_id)
        if not is_authorized(
            agent["name"],
            agent.get("role", ""),
            "submit_task",
            ctx.task_context.task_type,
        ):
            ctx.warning = "unauthorized"
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="unauthorized",
                    context_id=ctx.uuid,
                    detail={"agent": agent["name"], "action": "submit_task"},
                )
            )
            ctx.audit_trace.append(log_id)
            return False
        if (
            ctx.max_tokens
            and contract.max_tokens
            and ctx.max_tokens > contract.max_tokens
        ):
            ctx.warning = "token limit exceeded"
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="token_limit_exceeded",
                    context_id=ctx.uuid,
                    detail={"agent": agent["name"], "max": contract.max_tokens},
                )
            )
            ctx.audit_trace.append(log_id)
            return False
        return True

    def _skills_allowed(self, agent: dict[str, Any], ctx: ModelContext) -> bool:
        profile = AgentIdentity.load(agent["name"])
        allowed = match_agent_to_task(profile, ctx.required_skills or [])
        if not allowed:
            ctx.warning = "missing_skills"
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="skill_check_failed",
                    context_id=ctx.uuid,
                    detail={"agent": agent["name"], "skills": ctx.required_skills},
                )
            )
            ctx.audit_trace.append(log_id)
        return allowed

    def _endorsement_allowed(self, agent: dict[str, Any], ctx: ModelContext) -> bool:
        if not ctx.require_endorsement:
            return True
        from core.trust_circle import is_trusted_for

        if is_trusted_for(agent["name"], agent.get("role", "")):
            return True
        ctx.warning = "endorsement_missing"
        log_id = self.audit.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="dispatcher",
                action="endorsement_missing",
                context_id=ctx.uuid,
                detail={"agent": agent["name"], "role": agent.get("role")},
            )
        )
        ctx.audit_trace.append(log_id)
        return False

    def _prepare_context(
        self,
        task: TaskContext,
        session_id: str | None,
        task_value: float | None,
        max_tokens: int | None,
        priority: int | None = None,
        deadline: str | None = None,
        required_skills: list[str] | None = None,
        enforce_certification: bool = False,
        require_endorsement: bool = False,
        mission_id: str | None = None,
        mission_step: int | None = None,
        mission_role: str | None = None,
    ) -> ModelContext:
        history: List[dict] = []
        memory: List[dict] = []
        token_spent = 0
        if session_id:
            history = self._fetch_history(session_id)
            task.preferences = task.preferences or {}
            task.preferences["history"] = history
            if history:
                memory = history[-1].get("memory", [])
            for h in history:
                token_spent += int(h.get("metrics", {}).get("tokens_used", 0))

        TOKENS_IN.labels("task_dispatcher").inc(
            len(str(task.description or "").split())
        )

        if mission_id:
            from core.missions import AgentMission

            mission = AgentMission.load(mission_id)
            if (
                mission
                and mission_step is not None
                and 0 <= mission_step < len(mission.steps)
            ):
                step = mission.steps[mission_step]
                required_skills = required_skills or step.get("skill_required")
                deadline = deadline or step.get("deadline")
                mission_role = mission_role or step.get("role")

        ctx = ModelContext(
            task=task.task_id,
            task_context=task,
            session_id=session_id,
            memory=memory,
            task_value=task_value,
            max_tokens=max_tokens,
            token_spent=token_spent,
            priority=priority,
            deadline=deadline,
            required_skills=required_skills,
            enforce_certification=enforce_certification,
            require_endorsement=require_endorsement,
            mission_id=mission_id,
            mission_step=mission_step,
            mission_role=mission_role,
        )
        log_id = self.audit.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="dispatcher",
                action="task_accepted",
                context_id=ctx.uuid,
                detail={"task_type": task.task_type},
            )
        )
        ctx.audit_trace.append(log_id)
        return ctx

    def _execute_context(
        self, ctx: ModelContext, mode: str, enforce_certification: bool = False
    ) -> ModelContext:
        agents = self._fetch_agents(ctx.task_context.task_type)
        agents = [a for a in agents if self._governance_allowed(a, ctx)]
        agents = [a for a in agents if self._endorsement_allowed(a, ctx)]
        if ctx.required_skills:
            agents = [a for a in agents if self._skills_allowed(a, ctx)]
            if enforce_certification and not agents:
                return ctx
        if not agents:
            ctx.warning = "no eligible agents"
            return ctx
        if ctx.task_value is not None:
            for a in agents:
                cost = a.get("estimated_cost_per_token", 0.0) or 1e-6
                a["_score"] = ctx.task_value / cost
            agents.sort(key=lambda a: (-a["_score"], a.get("load_factor", 0)))

        if ctx.max_tokens is not None and ctx.token_spent >= ctx.max_tokens:
            ctx.warning = "budget exceeded"
            return ctx

        if mode == "single":
            agent = agents[0] if agents else None
            ctx.agent_selection = agent["id"] if agent else None
            if agent:
                log_id = self.audit.write(
                    AuditEntry(
                        timestamp=datetime.utcnow().isoformat(),
                        actor="dispatcher",
                        action="agent_selected",
                        context_id=ctx.uuid,
                        detail={"agent": agent["name"]},
                    )
                )
                ctx.audit_trace.append(log_id)
            if agent:
                self._apply_role_limits(ctx, agent.get("role", ""))
                arc = self._run_agent(agent, ctx)
                ctx.agents.append(arc)
                ctx.result = arc.result
                ctx.metrics = arc.metrics
                if arc.metrics:
                    limit = ctx.applied_limits.get("max_tokens", ctx.max_tokens or 0)
                    update_trust_usage(
                        agent["name"], int(arc.metrics.get("tokens_used", 0)), limit
                    )
        elif mode == "coalition":
            coalition = self._init_coalition(
                ctx.task_context.description or "",
                [a["id"] for a in agents],
            )
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="coalition_created",
                    context_id=ctx.uuid,
                    detail={"coalition": coalition.get("id")},
                )
            )
            ctx.audit_trace.append(log_id)
            ctx.task_context.preferences = ctx.task_context.preferences or {}
            ctx.task_context.preferences["coalition_id"] = coalition.get("id")
            for a in agents:
                self._assign_subtask(
                    coalition.get("id"), ctx.task_context.description or "", a["id"]
                )
            ctx.agents = [
                AgentRunContext(agent_id=a["id"], role=a.get("role"), url=a.get("url"))
                for a in agents
            ]
            for arc in ctx.agents:
                self._apply_role_limits(ctx, arc.role or "")
            ctx = self._send_to_coordinator(ctx, "parallel")
        else:
            ctx.agents = [
                AgentRunContext(agent_id=a["id"], role=a.get("role"), url=a.get("url"))
                for a in agents
            ]
            for arc in ctx.agents:
                self._apply_role_limits(ctx, arc.role or "")
            ctx = self._send_to_coordinator(ctx, mode)

        if ctx.metrics:
            ctx.token_spent += int(ctx.metrics.get("tokens_used", 0))
        if ctx.max_tokens is not None and ctx.token_spent > ctx.max_tokens:
            ctx.warning = "budget exceeded"
        TASKS_PROCESSED.labels("task_dispatcher").inc()
        tokens = ctx.metrics.get("tokens_used", 0) if ctx.metrics else 0
        TOKENS_OUT.labels("task_dispatcher").inc(tokens)
        if ctx.mission_id is not None:
            self._record_mission_progress(ctx)
        return ctx

    def dispatch_task(
        self,
        task: TaskContext,
        session_id: str | None = None,
        mode: str = "single",
        task_value: float | None = None,
        max_tokens: int | None = None,
        priority: int | None = None,
        deadline: str | None = None,
        required_skills: list[str] | None = None,
        enforce_certification: bool = False,
        require_endorsement: bool = False,
        mission_id: str | None = None,
        mission_step: int | None = None,
        mission_role: str | None = None,
    ) -> ModelContext:
        """Select agents and forward the ModelContext."""
        ctx = self._prepare_context(
            task,
            session_id,
            task_value,
            max_tokens,
            priority,
            deadline,
            required_skills,
            enforce_certification,
            require_endorsement,
            mission_id,
            mission_step,
            mission_role,
        )
        ctx.dispatch_state = "running"
        ctx = self._execute_context(ctx, mode, enforce_certification)
        ctx.dispatch_state = "completed"
        return ctx

    def enqueue_task(
        self,
        task: TaskContext,
        session_id: str | None = None,
        mode: str = "single",
        task_value: float | None = None,
        max_tokens: int | None = None,
        priority: int | None = None,
        deadline: str | None = None,
        required_skills: list[str] | None = None,
        enforce_certification: bool = False,
        require_endorsement: bool = False,
        mission_id: str | None = None,
        mission_step: int | None = None,
        mission_role: str | None = None,
    ) -> ModelContext:
        ctx = self._prepare_context(
            task,
            session_id,
            task_value,
            max_tokens,
            priority,
            deadline,
            required_skills,
            enforce_certification,
            require_endorsement,
            mission_id,
            mission_step,
            mission_role,
        )
        ctx.dispatch_state = "queued"
        self.queue.enqueue(ctx)
        return ctx

    def process_queue_once(self, mode: str = "single") -> ModelContext | None:
        self.queue.expire_old_tasks()
        ctx = self.queue.dequeue()
        if not ctx:
            return None
        ctx = self._execute_context(ctx, mode, ctx.enforce_certification)
        ctx.dispatch_state = "completed"
        return ctx

    def _fetch_agents(self, capability: str) -> list[dict[str, Any]]:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.registry_url}/agents")
                resp.raise_for_status()
                agents = resp.json().get("agents", [])
                candidates = [
                    a
                    for a in agents
                    if capability in a.get("capabilities", [])
                    or capability in a.get("skills", [])
                ]
                candidates.sort(
                    key=lambda a: (
                        a.get("load_factor", 0),
                        a.get("estimated_cost_per_token", 0),
                        a.get("avg_response_time", 0),
                    )
                )
                return candidates
        except Exception:
            return []

    def _fetch_history(self, session_id: str) -> list[dict]:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.session_url}/context/{session_id}")
                resp.raise_for_status()
                return resp.json().get("context", [])
        except Exception:
            return []

    def _run_agent(self, agent: dict[str, Any], ctx: ModelContext) -> AgentRunContext:
        """Call the worker's /run endpoint and return AgentRunContext."""
        start = time.perf_counter()
        contract = AgentContract.load(agent["name"])
        send_ctx = redact_context(ctx, contract.max_access_level)
        send_ctx = filter_permissions(send_ctx, agent.get("role", ""))
        if send_ctx.metrics and send_ctx.metrics.get("context_redacted_fields"):
            self.log.info(
                "context_redacted",
                agent=agent["name"],
                fields=send_ctx.metrics["context_redacted_fields"],
            )
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="context_redacted",
                    context_id=ctx.uuid,
                    detail={
                        "agent": agent["name"],
                        "fields": send_ctx.metrics["context_redacted_fields"],
                    },
                )
            )
            ctx.audit_trace.append(log_id)
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{agent['url'].rstrip('/')}/run",
                    json=send_ctx.model_dump(),
                    timeout=10,
                )
                resp.raise_for_status()
                data = ModelContext(**resp.json())
                verify = (
                    os.getenv("DISABLE_SIGNATURE_VALIDATION", "false").lower() != "true"
                )
                valid = True
                if verify:
                    if data.signed_by and data.signature:
                        payload = data.model_dump(exclude={"signature", "signed_by"})
                        valid = verify_signature(
                            data.signed_by, payload, data.signature
                        )
                    else:
                        valid = False
                    if not valid:
                        log_id = self.audit.write(
                            AuditEntry(
                                timestamp=datetime.utcnow().isoformat(),
                                actor="dispatcher",
                                action="signature_invalid",
                                context_id=ctx.uuid,
                                detail={"agent": agent["name"]},
                            )
                        )
                        ctx.audit_trace.append(log_id)
                        if contract.require_signature:
                            ctx.warning = "missing_signature"
                ctx.signed_by = data.signed_by
                ctx.signature = data.signature
                arc = AgentRunContext(
                    agent_id=agent["id"],
                    role=agent.get("role"),
                    url=agent.get("url"),
                    result=data.result,
                    metrics=data.metrics,
                )
        except Exception:
            arc = AgentRunContext(
                agent_id=agent["id"], role=agent.get("role"), url=agent.get("url")
            )
        duration = time.perf_counter() - start
        self._update_status(agent["name"], duration)
        return arc

    def _send_to_coordinator(self, ctx: ModelContext, mode: str) -> ModelContext:
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.coordinator_url}/coordinate",
                    json={"context": ctx.model_dump(), "mode": mode},
                    timeout=10,
                )
                resp.raise_for_status()
                return ModelContext(**resp.json())
        except Exception:
            return ctx

    def _init_coalition(self, goal: str, members: List[str]) -> dict:
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.coalition_url}/coalition/init",
                    json={
                        "goal": goal,
                        "leader": members[0] if members else "",
                        "members": members,
                    },
                    timeout=5,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {
                "id": "local",
                "goal": goal,
                "leader": members[0] if members else "",
                "members": members,
                "strategy": "parallel-expert",
                "subtasks": [],
            }

    def _assign_subtask(self, coalition_id: str, title: str, assigned_to: str) -> None:
        try:
            with httpx.Client() as client:
                client.post(
                    f"{self.coalition_url}/coalition/{coalition_id}/assign",
                    json={"title": title, "assigned_to": assigned_to},
                    timeout=5,
                )
        except Exception:
            pass

    def _record_mission_progress(self, ctx: ModelContext) -> None:
        from datetime import datetime
        from core.missions import AgentMission
        from core.rewards import grant_rewards

        mission = AgentMission.load(ctx.mission_id or "")
        if not mission or not ctx.agents:
            return
        agent_id = ctx.agents[0].agent_id
        profile = AgentIdentity.load(agent_id)
        progress = profile.mission_progress.get(
            ctx.mission_id, {"step": 0, "status": "in_progress"}
        )
        if ctx.mission_step is not None and ctx.mission_step + 1 > progress.get(
            "step", 0
        ):
            progress["step"] = ctx.mission_step + 1
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="mission_step_completed",
                    context_id=ctx.uuid,
                    detail={
                        "mission": ctx.mission_id,
                        "agent": agent_id,
                        "step": ctx.mission_step,
                    },
                )
            )
            ctx.audit_trace.append(log_id)
        if progress.get("step", 0) >= len(mission.steps):
            progress["status"] = "complete"
            if ctx.mission_id in profile.active_missions:
                profile.active_missions.remove(ctx.mission_id)
            grant_rewards(agent_id, mission.rewards)
            log_id = self.audit.write(
                AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    actor="dispatcher",
                    action="mission_completed",
                    context_id=ctx.uuid,
                    detail={"mission": ctx.mission_id, "agent": agent_id},
                )
            )
            ctx.audit_trace.append(log_id)
        else:
            if ctx.mission_id not in profile.active_missions:
                profile.active_missions.append(ctx.mission_id)
        profile.mission_progress[ctx.mission_id] = progress
        profile.save()

    def _update_status(self, agent_name: str, duration: float) -> None:
        """Send runtime metrics to the registry."""
        payload = {
            "busy": False,
            "tasks_in_progress": 0,
            "last_response_duration": duration,
        }
        try:
            with httpx.Client() as client:
                client.post(
                    f"{self.registry_url}/agent_status/{agent_name}",
                    json=payload,
                    timeout=5,
                )
        except Exception:
            pass
