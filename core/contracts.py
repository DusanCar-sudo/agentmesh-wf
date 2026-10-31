"""
AgentMesh · core/contracts.py
Execution Contracts — typed, bounded agent call definitions.

Key research finding (Tsinghua 2026):
  Fuzzy "did it finish?" checks → verifier agents → performance DROPS (-8.4).
  Hard boolean completion conditions → cheaper, more reliable.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# ── Contract definition ───────────────────────────────────────────────────────

@dataclass
class ExecutionContract:
    """
    Defines a single bounded agent task.
    Think of this as a function signature for an agent call.
    """
    # Identity
    task_id: str
    agent_id: str
    description: str

    # Budget (the research shows tight budgets = same quality, 13x less cost)
    max_tokens: int = 4000
    max_tool_calls: int = 12
    timeout_seconds: int = 300

    # Permissions — agent can ONLY call tools listed here
    allowed_tools: list[str] = field(default_factory=list)

    # Hard completion gates — ALL must be True for task to count as done
    # Example: ["output_file_exists", "no_error_flag", "summary_written"]
    completion_conditions: list[str] = field(default_factory=list)

    # Where the agent writes its result
    output_path: str = ""

    # What the output must contain (keys checked in JSON result)
    required_output_keys: list[str] = field(default_factory=list)

    def output_file(self) -> Path:
        return Path(self.output_path) if self.output_path else Path(f"workspace/{self.agent_id}/result.json")

    def to_prompt_block(self) -> str:
        """Serialise contract into a system-prompt block the agent reads."""
        lines = [
            f"TASK_ID: {self.task_id}",
            f"DESCRIPTION: {self.description}",
            f"MAX_TOKENS: {self.max_tokens}",
            f"MAX_TOOL_CALLS: {self.max_tool_calls}",
            f"ALLOWED_TOOLS: {', '.join(self.allowed_tools) or 'none'}",
            f"OUTPUT_PATH: {self.output_file()}",
            f"REQUIRED_OUTPUT_KEYS: {', '.join(self.required_output_keys) or 'none'}",
            "COMPLETION_CONDITIONS:",
        ]
        for c in self.completion_conditions:
            lines.append(f"  - {c}")
        return "\n".join(lines)

    def save(self, base_dir: str = "workspace") -> Path:
        path = Path(base_dir) / self.agent_id / "contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, agent_id: str, base_dir: str = "workspace") -> "ExecutionContract":
        path = Path(base_dir) / agent_id / "contract.json"
        data = json.loads(path.read_text())
        return cls(**data)


# ── Contract result ───────────────────────────────────────────────────────────

@dataclass
class ContractResult:
    task_id: str
    agent_id: str
    success: bool
    output: dict
    tokens_used: int = 0
    tool_calls_made: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    failed_conditions: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save(self, base_dir: str = "workspace") -> Path:
        path = Path(base_dir) / self.agent_id / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))
        return path

    @classmethod
    def load(cls, agent_id: str, base_dir: str = "workspace") -> "ContractResult":
        path = Path(base_dir) / agent_id / "result.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(**data)


# ── Condition checker ─────────────────────────────────────────────────────────

class ConditionChecker:
    """
    Evaluates completion_conditions against actual agent output.
    Replaces expensive verifier agents with cheap boolean checks.
    """

    def check_all(self, contract: ExecutionContract, result_data: dict) -> tuple[bool, list[str]]:
        """
        Returns (all_passed, list_of_failed_conditions).
        """
        failed = []
        for condition in contract.completion_conditions:
            if not self._check_one(condition, contract, result_data):
                failed.append(condition)
        return len(failed) == 0, failed

    def _check_one(self, condition: str, contract: ExecutionContract, data: dict) -> bool:
        c = condition.strip().lower()

        if c == "output_file_exists":
            return contract.output_file().exists()

        if c == "no_error_flag":
            return not data.get("error") and not data.get("failed")

        if c == "summary_written":
            return bool(data.get("summary", "").strip())

        if c == "required_keys_present":
            return all(k in data for k in contract.required_output_keys)

        if c.startswith("key_exists:"):
            key = c.split(":", 1)[1].strip()
            return key in data

        if c.startswith("min_length:"):
            # format: min_length:field_name:100
            parts = c.split(":")
            if len(parts) == 3:
                _, fname, min_len = parts
                return len(str(data.get(fname, ""))) >= int(min_len)

        # Unknown condition — log and pass (don't block on unknown rules)
        print(f"[ConditionChecker] Unknown condition '{condition}' — skipping")
        return True


# ── Factory helpers ───────────────────────────────────────────────────────────

def research_contract(task_id: str, query: str, output_dir: str = "workspace") -> ExecutionContract:
    """Preset contract for a research/RAG sub-agent."""
    return ExecutionContract(
        task_id=task_id,
        agent_id="agent_research",
        description=f"Research and summarise: {query}",
        max_tokens=3000,
        max_tool_calls=8,
        allowed_tools=["web_search", "read_file"],
        completion_conditions=["no_error_flag", "summary_written", "required_keys_present"],
        output_path=f"{output_dir}/agent_research/result.json",
        required_output_keys=["summary", "sources"],
    )

def code_contract(task_id: str, spec: str, output_dir: str = "workspace") -> ExecutionContract:
    """Preset contract for a code/dev sub-agent."""
    return ExecutionContract(
        task_id=task_id,
        agent_id="agent_code",
        description=f"Write code for: {spec}",
        max_tokens=4000,
        max_tool_calls=10,
        allowed_tools=["run_python", "read_file", "write_file"],
        completion_conditions=["output_file_exists", "no_error_flag", "required_keys_present"],
        output_path=f"{output_dir}/agent_code/result.json",
        required_output_keys=["code", "explanation"],
    )

def writer_contract(task_id: str, topic: str, output_dir: str = "workspace") -> ExecutionContract:
    """Preset contract for a content/writer sub-agent."""
    return ExecutionContract(
        task_id=task_id,
        agent_id="agent_writer",
        description=f"Write content about: {topic}",
        max_tokens=3500,
        max_tool_calls=4,
        allowed_tools=["read_file"],
        completion_conditions=["no_error_flag", "summary_written", "required_keys_present"],
        output_path=f"{output_dir}/agent_writer/result.json",
        required_output_keys=["content", "title"],
    )


def data_contract(task_id: str, spec: str, output_dir: str = "workspace") -> ExecutionContract:
    """Preset contract for a data analysis sub-agent."""
    return ExecutionContract(
        task_id=task_id,
        agent_id="agent_data",
        description=f"Analyze data for: {spec}",
        max_tokens=3000,
        max_tool_calls=8,
        allowed_tools=["run_python", "read_file"],
        completion_conditions=["no_error_flag", "required_keys_present"],
        output_path=f"{output_dir}/agent_data/result.json",
        required_output_keys=["analysis", "statistics"],
    )


def summary_contract(task_id: str, topic: str, output_dir: str = "workspace") -> ExecutionContract:
    """Preset contract for a summarization sub-agent."""
    return ExecutionContract(
        task_id=task_id,
        agent_id="agent_summary",
        description=f"Summarize: {topic}",
        max_tokens=2000,
        max_tool_calls=4,
        allowed_tools=["read_file"],
        completion_conditions=["no_error_flag", "summary_written", "required_keys_present"],
        output_path=f"{output_dir}/agent_summary/result.json",
        required_output_keys=["summary", "key_points"],
    )
