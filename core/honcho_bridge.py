"""
AgentMesh · core/honcho_bridge.py
Honcho Integration — AI-native peer modeling and dialectic memory.

Architecture with Honcho:
  Honcho handles:  WHO each agent is — learned patterns, behavior models,
                   dialectic reasoning, cross-session identity
  memory.py keeps: WHAT happened — raw episodes, skill stats, failure traces

Two-layer memory system:
  ┌─────────────────────────────────────────────────────┐
  │  Honcho (cloud)                                     │
  │  • Peer models for each agent + orchestrator        │
  │  • Session messages with reasoning pipeline         │
  │  • context() — synthesised agent profile            │
  │  • chat()    — "what patterns does agent_code show?"│
  └─────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────┐
  │  SQLite memory.py (local)                           │
  │  • Raw verbatim traces (never summarised)           │
  │  • Episode history + skill stats                    │
  │  • Lessons + acceptance-gated evolution             │
  └─────────────────────────────────────────────────────┘

Usage:
    bridge = HonchoBridge(api_key="...", workspace_id="agentmesh")

    # Before task: enrich system prompt with agent's accumulated model
    ctx = bridge.get_agent_context("agent_research")
    enriched_prompt = base_prompt + ctx

    # After task: store what happened
    bridge.record_task(agent_id, task_desc, result_text, success)

    # Orchestrator asks about an agent
    insight = bridge.ask_about_agent("agent_code",
        "What types of tasks does this agent fail on?")
"""

from __future__ import annotations
import os
import time
import json
from typing import Optional
from pathlib import Path


class HonchoBridge:
    """
    Wraps Honcho SDK for AgentMesh.
    Each agent and the orchestrator become Honcho Peers.
    Sessions map 1:1 to task cycles.
    """

    WORKSPACE = "agentmesh"

    def __init__(
        self,
        api_key: str = None,
        workspace_id: str = None,
        routing_client=None,
    ):
        self.api_key = api_key or os.environ.get("HONCHO_API_KEY")
        if not self.api_key:
            raise HonchoError(
                "HONCHO_API_KEY not set. "
                "Get your key at https://app.honcho.dev → API KEYS"
            )
        self.workspace_id = workspace_id or os.environ.get(
            "HONCHO_WORKSPACE_ID", self.WORKSPACE
        )
        self._client = None
        self._peers: dict[str, object] = {}       # agent_id → Peer
        self._active_sessions: dict[str, object] = {}  # task_id → Session
        self._routing_client = routing_client      # LLM client for agent routing

    @property
    def client(self):
        """Lazy init — only connects when first used."""
        if self._client is None:
            from honcho import Honcho
            self._client = Honcho(
                api_key=self.api_key,
                workspace_id=self.workspace_id,
            )
        return self._client

    # ── Peer management ───────────────────────────────────────────────────────

    def get_peer(self, agent_id: str):
        """Get or create a Honcho Peer for an agent. Never raises — returns None on failure."""
        if agent_id not in self._peers:
            try:
                self._peers[agent_id] = self.client.peer(agent_id)
            except Exception as e:
                print(f"[HonchoBridge] get_peer warning ({agent_id}): {e}")
                return None
        return self._peers.get(agent_id)

    def register_agent(self, agent_id: str, role: str = "", description: str = "") -> None:
        """Register agent as Honcho Peer. Silently skips on any error."""
        try:
            peer = self.get_peer(agent_id)
            if peer is None:
                return
            peer.set_metadata({
                "role": role,
                "description": description,
                "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "framework": "AgentMesh",
            })
        except Exception as e:
            print(f"[HonchoBridge] register_agent warning ({agent_id}): {e}")

    # ── Session management ────────────────────────────────────────────────────

    def start_session(self, task_id: str, agent_ids: list[str]):
        """Open Honcho session. Returns None silently if unavailable."""
        try:
            session = self.client.session(
                task_id,
                peers=agent_ids,
                metadata={"task_id": task_id, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            )
            self._active_sessions[task_id] = session
            return session
        except Exception:
            return None

    def end_session(self, task_id: str) -> None:
        self._active_sessions.pop(task_id, None)

    # ── Message storage ───────────────────────────────────────────────────────

    def record_task(
        self,
        agent_id: str,
        task_description: str,
        result: str,
        success: bool,
        task_id: str = None,
        tokens_used: int = 0,
    ) -> None:
        """
        Store a completed task as a message exchange in Honcho.
        Honcho's reasoning pipeline will asynchronously build a model
        of this agent's behaviour from these messages.
        """
        task_id = task_id or f"task_{int(time.time())}"
        session_id = f"{agent_id}_{task_id}"

        try:
            session = self.client.session(
                session_id,
                peers=[agent_id],
                metadata={
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "success": success,
                    "tokens_used": tokens_used,
                },
            )
            # Honcho v2: MessageCreateParams has no role field
            # Send task + result as two separate peer messages
            outcome = "SUCCESS" if success else "FAILED"
            res_text = result[:500] if result else "(no output)"
            session.add_messages([
                {"peer_id": agent_id, "content": f"TASK: {task_description}"},
                {"peer_id": agent_id, "content": f"{outcome}: {res_text}"},
            ])
        except Exception as e:
            print(f"[HonchoBridge] record_task warning for {agent_id}: {e}")

    def record_tool_use(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        args: dict,
        result: str,
        success: bool,
    ) -> None:
        """Log individual tool calls within a task session."""
        try:
            session = self._active_sessions.get(session_id)
            if not session:
                return
            session.add_messages([{
                "peer_id": agent_id,
                "content": (
                    f"[TOOL:{tool_name}] args={json.dumps(args)[:200]} "
                    f"→ {'OK' if success else 'FAIL'}: {result[:200]}"
                ),
            }])
        except Exception as e:
            pass  # Tool logging is best-effort

    # ── Context retrieval ─────────────────────────────────────────────────────

    def get_agent_context(self, agent_id: str, search_query: str = None) -> str:
        """
        Get Honcho's synthesised model of this agent.
        Returns a text block to inject into the agent's system prompt.

        This is the key call — Honcho assembles:
          - Agent representation (patterns, behaviors, tendencies)
          - Session history summary
          - Dialectic reasoning about current state
        """
        try:
            peer = self.get_peer(agent_id)
            ctx = peer.context(search_query=search_query)

            if not ctx:
                return ""

            # Format into a clean prompt block
            parts = []
            if ctx.representation:
                parts.append(f"AGENT PROFILE:\n{ctx.representation}")
            if ctx.peer_card:
                card = ctx.peer_card
                card_text = "\n".join(card) if isinstance(card, list) else str(card)
                parts.append(f"AGENT CARD:\n{card_text}")

            return "\n\n".join(parts) if parts else str(ctx)

        except Exception as e:
            print(f"[HonchoBridge] get_agent_context warning for {agent_id}: {e}")
            return ""

    def get_session_context(
        self,
        task_id: str,
        agent_id: str,
        max_tokens: int = 1000,
    ) -> str:
        """
        Get context scoped to the current session.
        Used mid-task to maintain continuity.
        """
        try:
            session = self._active_sessions.get(task_id)
            if not session:
                return ""
            ctx = session.context(peer_target=agent_id, tokens=max_tokens)
            return str(ctx) if ctx else ""
        except Exception as e:
            return ""

    # ── Dialectic queries ─────────────────────────────────────────────────────

    def ask_about_agent(
        self,
        agent_id: str,
        question: str,
        reasoning_level: str = "medium",
    ) -> str:
        """
        Ask Honcho a natural language question about an agent's patterns.
        This is the meta-harness use case — the orchestrator queries what
        it knows about each sub-agent before routing tasks.

        Examples:
            "What types of tasks does this agent struggle with?"
            "What is this agent's typical token usage pattern?"
            "Does this agent handle API errors well?"
        """
        try:
            peer = self.get_peer(agent_id)
            answer = peer.chat(question, reasoning_level=reasoning_level)
            return answer or ""
        except Exception as e:
            print(f"[HonchoBridge] ask_about_agent warning for {agent_id}: {e}")
            return ""

    def compare_agents(self, agent_ids: list[str], question: str) -> dict[str, str]:
        """
        Ask the same question about multiple agents.
        Orchestrator uses this for smart task routing.

        Example:
            results = bridge.compare_agents(
                ["agent_research", "agent_code"],
                "What is this agent best suited for?"
            )
        """
        return {aid: self.ask_about_agent(aid, question) for aid in agent_ids}

    def best_agent_for_task(self, agent_ids: list[str], task_description: str) -> str:
        """
        Ask Honcho which agent is best suited for a given task.
        Returns the agent_id Honcho recommends.
        """
        profiles = {}
        for aid in agent_ids:
            ctx = self.get_agent_context(aid, search_query=task_description)
            if ctx:
                profiles[aid] = ctx

        if not profiles:
            return agent_ids[0]

        prompt = f"""Given these agent profiles and a task, return only the agent_id best suited.

TASK: {task_description}

AGENTS:
""" + "\n\n".join(f"[{aid}]:\n{profile}" for aid, profile in profiles.items()) + """

Return only the agent_id, nothing else."""

        if not self._routing_client:
            return agent_ids[0]

        try:
            response = self._routing_client.complete(prompt, max_tokens=30)
            response = response.strip().strip('"').strip("'")
            for aid in agent_ids:
                if aid in response:
                    return aid
        except Exception as e:
            print(f"[HonchoBridge] best_agent_for_task routing failed: {e}")

        return agent_ids[0]

    # ── Search ────────────────────────────────────────────────────────────────

    def search_memory(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search across all stored messages in the workspace."""
        try:
            results = self.client.search(query=query, limit=limit)
            return [
                {
                    "content": r.content if hasattr(r, "content") else str(r),
                    "peer_id": getattr(r, "peer_id", ""),
                    "session_id": getattr(r, "session_id", ""),
                }
                for r in (results.items if hasattr(results, "items") else [results])
            ]
        except Exception as e:
            print(f"[HonchoBridge] search_memory warning: {e}")
            return []

    # ── Combined context for Step 3 ───────────────────────────────────────────

    def enrich_system_prompt(
        self,
        agent_id: str,
        base_prompt: str,
        task_description: str = "",
    ) -> str:
        """
        Full enrichment pipeline for an agent's system prompt:
          1. Honcho agent profile (who this agent is, learned patterns)
          2. Task-relevant context from Honcho (what matters for THIS task)

        Called in BaseAgent.run() before every task.
        Empty sections are silently omitted.
        """
        parts = [base_prompt]

        # Agent profile (who they are across all sessions)
        profile = self.get_agent_context(
            agent_id,
            search_query=task_description or None,
        )
        if profile:
            parts.append(f"--- HONCHO AGENT CONTEXT ---\n{profile}")

        return "\n\n".join(parts)

    def is_available(self) -> bool:
        """Quick connectivity check."""
        try:
            _ = self.client  # triggers lazy init + workspace ensure
            return True
        except Exception:
            return False


class HonchoError(Exception):
    pass


# ── Null bridge (fallback when Honcho not configured) ─────────────────────────

class NullHonchoBridge:
    """
    Drop-in replacement when HONCHO_API_KEY is not set.
    All methods return empty strings / no-ops.
    The agent still works — just without Honcho's dialectic enrichment.
    """

    def register_agent(self, *a, **kw): pass
    def start_session(self, *a, **kw): return None
    def end_session(self, *a, **kw): pass
    def record_task(self, *a, **kw): pass
    def record_tool_use(self, *a, **kw): pass
    def get_agent_context(self, *a, **kw) -> str: return ""
    def get_session_context(self, *a, **kw) -> str: return ""
    def ask_about_agent(self, *a, **kw) -> str: return ""
    def compare_agents(self, agent_ids, *a, **kw) -> dict: return {aid: "" for aid in agent_ids}
    def best_agent_for_task(self, agent_ids, *a, **kw) -> str: return agent_ids[0]
    def search_memory(self, *a, **kw) -> list: return []
    def enrich_system_prompt(self, agent_id, base_prompt, *a, **kw) -> str: return base_prompt
    def is_available(self) -> bool: return False


def get_honcho_bridge(
    api_key: str = None,
    workspace_id: str = None,
    routing_client=None,
) -> "HonchoBridge | NullHonchoBridge":
    """
    Factory — returns real bridge if key available, null bridge otherwise.
    This lets all agent code be written against the same interface.
    """
    key = api_key or os.environ.get("HONCHO_API_KEY", "")
    if key:
        return HonchoBridge(api_key=key, workspace_id=workspace_id, routing_client=routing_client)
    print("[HonchoBridge] HONCHO_API_KEY not set — using NullBridge (no peer modeling)")
    return NullHonchoBridge()
