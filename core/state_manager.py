"""
AgentMesh · core/state_manager.py
File-Backed State — agent working memory lives on disk, not in context window.

Key research finding (Tsinghua 2026):
  Externalise memory to path-addressable files.
  Survives truncation, restarts, and delegation handoffs.
  Orchestrator can check progress without querying the agent.
"""

from __future__ import annotations
import json
import time
import fcntl
from pathlib import Path
from typing import Any, Optional
from enum import Enum


class AgentStatus(str, Enum):
    IDLE       = "idle"
    RUNNING    = "running"
    WAITING    = "waiting"       # waiting for another agent's output
    DONE       = "done"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


class FileBackedState:
    """
    Each agent owns a directory: workspace/{agent_id}/
    ├── state.json       — live working state (read by orchestrator)
    ├── contract.json    — the task this agent is executing
    ├── result.json      — final output (written on completion)
    ├── checkpoint.json  — mid-task progress (resume after crash)
    └── trace.jsonl      — append-only raw execution log
    """

    def __init__(self, agent_id: str, base_dir: str = "workspace"):
        self.agent_id = agent_id
        self.base = Path(base_dir) / agent_id
        self.base.mkdir(parents=True, exist_ok=True)

        self._state_path      = self.base / "state.json"
        self._checkpoint_path = self.base / "checkpoint.json"
        self._trace_path      = self.base / "trace.jsonl"
        self._result_path     = self.base / "result.json"

        # Initialise state file if new agent
        if not self._state_path.exists():
            self._write_state({
                "agent_id": agent_id,
                "status": AgentStatus.IDLE,
                "task_id": None,
                "step": 0,
                "total_steps": 0,
                "current_action": "",
                "tokens_used": 0,
                "tool_calls": 0,
                "started_at": None,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "context": {},      # arbitrary agent-specific data
            })

    # ── State read/write ──────────────────────────────────────────────────────

    def get(self) -> dict:
        """Read current state. Orchestrator calls this to check progress."""
        try:
            return json.loads(self._state_path.read_text())
        except Exception:
            return {}

    def update(self, **kwargs) -> None:
        """Update individual state fields atomically."""
        state = self.get()
        state.update(kwargs)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_state(state)

    def set_status(self, status: AgentStatus, action: str = "") -> None:
        self.update(status=status, current_action=action)

    def set_running(self, task_id: str, total_steps: int = 0) -> None:
        self.update(
            status=AgentStatus.RUNNING,
            task_id=task_id,
            step=0,
            total_steps=total_steps,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def advance_step(self, action: str = "") -> None:
        state = self.get()
        self.update(step=state.get("step", 0) + 1, current_action=action)

    def inc_tokens(self, n: int) -> None:
        state = self.get()
        self.update(tokens_used=state.get("tokens_used", 0) + n)

    def inc_tool_calls(self) -> None:
        state = self.get()
        self.update(tool_calls=state.get("tool_calls", 0) + 1)

    def set_context(self, key: str, value: Any) -> None:
        """Store arbitrary agent-specific data (won't bloat context window)."""
        state = self.get()
        ctx = state.get("context", {})
        ctx[key] = value
        self.update(context=ctx)

    def get_context(self, key: str, default=None) -> Any:
        return self.get().get("context", {}).get(key, default)

    # ── Checkpoint (crash recovery) ───────────────────────────────────────────

    def checkpoint(self, data: dict) -> None:
        """Save mid-task progress. Called after each significant step."""
        payload = {
            **data,
            "agent_id": self.agent_id,
            "checkpoint_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._checkpoint_path.write_text(json.dumps(payload, indent=2))

    def load_checkpoint(self) -> Optional[dict]:
        """Resume from last checkpoint after a crash/restart."""
        if self._checkpoint_path.exists():
            try:
                return json.loads(self._checkpoint_path.read_text())
            except Exception:
                return None
        return None

    def clear_checkpoint(self) -> None:
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

    # ── Result (final output) ─────────────────────────────────────────────────

    def write_result(self, data: dict) -> Path:
        """Write the final task result. Sets status to DONE."""
        result = {
            **data,
            "agent_id": self.agent_id,
            "task_id": self.get().get("task_id"),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._result_path.write_text(json.dumps(result, indent=2))
        self.set_status(AgentStatus.DONE, "completed")
        self.clear_checkpoint()
        return self._result_path

    def read_result(self) -> Optional[dict]:
        if self._result_path.exists():
            return json.loads(self._result_path.read_text())
        return None

    def result_exists(self) -> bool:
        return self._result_path.exists()

    # ── Trace log (raw, verbatim — NEVER summarise) ───────────────────────────

    def trace(self, event: str, data: dict = None) -> None:
        """
        Append-only raw execution log.
        Research finding: accuracy drops from 50% to 34.6% if you replace
        raw traces with summaries. Keep EVERYTHING.
        """
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "agent_id": self.agent_id,
            "event": event,
            **(data or {}),
        }
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_traces(self, last_n: int = 50) -> list[dict]:
        """Return last N trace entries (for orchestrator self-evolution loop)."""
        if not self._trace_path.exists():
            return []
        lines = self._trace_path.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-last_n:]]

    def get_failed_traces(self) -> list[dict]:
        """Return only failure events — key input for harness self-evolution."""
        return [t for t in self.get_traces(200) if "fail" in t.get("event", "").lower()
                or "error" in t.get("event", "").lower()]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def status(self) -> AgentStatus:
        return AgentStatus(self.get().get("status", "idle"))

    @property
    def is_done(self) -> bool:
        return self.status == AgentStatus.DONE

    @property
    def is_running(self) -> bool:
        return self.status == AgentStatus.RUNNING

    def reset(self) -> None:
        """Reset agent to idle — keeps trace log intact."""
        self.update(
            status=AgentStatus.IDLE,
            task_id=None,
            step=0,
            tokens_used=0,
            tool_calls=0,
            current_action="",
            context={},
        )
        self.clear_checkpoint()
        if self._result_path.exists():
            self._result_path.unlink()

    def _write_state(self, state: dict) -> None:
        """Atomic write via temp file."""
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self._state_path)


# ── Orchestrator state (tracks all agents) ───────────────────────────────────

class OrchestratorState:
    """Top-level state for the orchestrator PC."""

    def __init__(self, base_dir: str = "workspace"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._path = self.base / "_orchestrator.json"

        if not self._path.exists():
            self._write({
                "status": "idle",
                "current_goal": "",
                "tasks": {},         # task_id → {agent_id, status, created_at}
                "iteration": 0,
                "total_tokens": 0,
                "started_at": None,
            })

    def get(self) -> dict:
        return json.loads(self._path.read_text())

    def register_task(self, task_id: str, agent_id: str) -> None:
        state = self.get()
        state["tasks"][task_id] = {
            "agent_id": agent_id,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._write(state)

    def mark_task_done(self, task_id: str) -> None:
        state = self.get()
        if task_id in state["tasks"]:
            state["tasks"][task_id]["status"] = "done"
            state["tasks"][task_id]["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write(state)

    def set_goal(self, goal: str) -> None:
        state = self.get()
        state["current_goal"] = goal
        state["status"] = "running"
        state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        state["iteration"] += 1
        self._write(state)

    def summary(self) -> str:
        s = self.get()
        tasks = s.get("tasks", {})
        done = sum(1 for t in tasks.values() if t["status"] == "done")
        return (f"Goal: {s['current_goal']} | "
                f"Tasks: {done}/{len(tasks)} done | "
                f"Iteration: {s['iteration']}")

    def _write(self, data: dict) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._path)
