"""
AgentMesh · orchestrator_lite.py
Lean orchestrator — rule-based routing, no LLM call for decomposition.

Token savings vs original:
  Original: 1 LLM call to decompose + up to 9 per agent + 1 learning + 1 harness = ~12+ calls
  Lite:     0 for routing + up to 3 per agent + 0 learning (batched) = ~3-4 calls per goal
"""

from __future__ import annotations
import json, time, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

# Load .env file (no external dependency)
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.split("#")[0].strip()
            if key:
                os.environ[key] = value

_load_env()

from core.contracts import ExecutionContract, ContractResult
from core.state_manager import OrchestratorState
from core.skill_registry import SkillRegistry
from core.hermes_client import HermesClient, Provider
from core.memory import AgentMemory
from core.learner import LearningCycle
from agents.base_agent_lite import BaseAgentLite


# ── Concrete sub-agents (lite) ───────────────────────────────────────────────

class ResearchAgentLite(BaseAgentLite):
    agent_id = "agent_research"
    system_prompt = (
        "You are a research agent. Find accurate information using web_search. "
        "Cite sources. Return JSON with 'summary' and 'sources'."
    )

class CodeAgentLite(BaseAgentLite):
    agent_id = "agent_code"
    system_prompt = (
        "You are a code agent. Write clean Python. "
        "Test with run_python before returning. Return JSON with 'code' and 'explanation'."
    )

class WriterAgentLite(BaseAgentLite):
    agent_id = "agent_writer"
    system_prompt = (
        "You are a content writer. Write clear, structured text. "
        "Read any provided files first. Return JSON with 'title' and 'content'."
    )

class FileAgentLite(BaseAgentLite):
    agent_id = "agent_file"
    system_prompt = (
        "You are a file management agent. Read, write, organise files. "
        "Return JSON with 'files_modified' and 'summary'."
    )

class DataAgentLite(BaseAgentLite):
    agent_id = "agent_data"
    system_prompt = (
        "You are a data analysis agent. Process datasets, compute statistics. "
        "Use run_python for computations. Return JSON with 'analysis' and 'insights'."
    )

class SummaryAgentLite(BaseAgentLite):
    agent_id = "agent_summary"
    system_prompt = (
        "You are a summarization agent. Condense content into key points. "
        "Return JSON with 'summary' and 'key_points'."
    )


# ── Keyword-based router ─────────────────────────────────────────────────────

ROUTE_TABLE: list[tuple[list[str], str]] = [
    # Priority order: first keyword match wins.
    # Writer before summary — "write a brief report" should hit writer.
    (["analyse", "analyze", "data", "dataset", "statistics", "csv", "metrics", "compute"],
     "agent_data"),
    (["code", "python", "script", "function", "class", "program", "debug", "fix", "implement"],
     "agent_code"),
    (["file", "organise", "organize", "directory", "rename", "move", "copy", "backup"],
     "agent_file"),
    (["write", "report", "document", "draft", "blog", "article", "email", "essay"],
     "agent_writer"),
    (["summarise", "summarize", "condense", "tldr", "digest", "key points"],
     "agent_summary"),
    (["research", "find", "search", "what is", "who is", "how does", "explain", "compare"],
     "agent_research"),
]


def route_goal_to_agents(goal: str) -> list[tuple[str, str]]:
    """
    Rule-based goal → agent routing. No LLM call.
    Returns list of (agent_id, task_description) tuples.
    """
    goal_lower = goal.lower()

    # Split compound goals
    clauses = _split_goal(goal)

    if len(clauses) <= 1:
        agent_id = _match_agent(goal_lower)
        return [(agent_id, goal)]
    else:
        tasks = []
        for clause in clauses:
            agent_id = _match_agent(clause.lower().strip())
            tasks.append((agent_id, clause.strip()))
        return tasks


def _split_goal(goal: str) -> list[str]:
    """Split compound goals into individual tasks."""
    action_verbs = {"research", "write", "code", "find", "analyse", "analyze",
                    "create", "build", "summarise", "summarize", "read", "save",
                    "search", "compute", "process", "fetch", "download"}
    for splitter in [" and then ", " and ", "; "]:
        if splitter in goal.lower():
            candidates = goal.split(splitter)
            valid_parts = []
            for c in candidates:
                words = set(c.lower().split())
                if words & action_verbs:
                    valid_parts.append(c)
            if len(valid_parts) >= 2:
                return valid_parts
    return [goal]


def _match_agent(text: str) -> str:
    for keywords, agent_id in ROUTE_TABLE:
        for kw in keywords:
            if kw in text:
                return agent_id
    return "agent_research"


# ── Task plan ─────────────────────────────────────────────────────────────────

class TaskPlanLite:
    def __init__(self, goal: str, tasks: list[ExecutionContract], parallel: list[list[str]] = None):
        self.goal = goal
        self.tasks = {t.task_id: t for t in tasks}
        self.parallel = parallel or [[t.task_id] for t in tasks]
        self.results: dict[str, ContractResult] = {}

    def all_done(self) -> bool:
        return all(tid in self.results for tid in self.tasks)


# ── Lite orchestrator ─────────────────────────────────────────────────────────

class OrchestratorLite:
    """
    Lean orchestrator. Rule-based routing, no LLM decomposition.
    Sub-agents use tool loop with max 3 iterations.
    Learning is batched, not per-task.
    """

    WORKSPACE = "workspace"

    def __init__(
        self,
        provider: Provider = Provider.DEEPSEEK,
        workspace: str = "workspace",
        stream: bool = False,
        enable_learning: bool = False,
        learning_batch_size: int = 5,
    ):
        self.workspace = workspace
        self.stream = stream
        self.enable_learning = enable_learning
        self.learning_batch_size = learning_batch_size
        Path(workspace).mkdir(exist_ok=True)

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

        self.client = HermesClient(
            provider=provider, api_key=api_key,
            model="deepseek-chat" if provider == Provider.DEEPSEEK else None,
        )
        self.registry = SkillRegistry("skills")
        self.memory = AgentMemory(f"{workspace}/memory.db")
        self.orch_state = OrchestratorState(workspace)

        self.cycle = None
        if enable_learning:
            self.cycle = LearningCycle(self.memory, self.client)

        self._agents = self._build_agents()
        self._tasks_since_learning = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, goal: str) -> dict:
        print(f"\n[OrchestratorLite] Goal: {goal}")
        print("=" * 60)

        self.orch_state.set_goal(goal)

        # 1. Route goal to agents (rule-based, zero LLM calls)
        routes = route_goal_to_agents(goal)
        print(f"[OrchestratorLite] Routed to {len(routes)} agent(s): {[r[0] for r in routes]}")

        # 2. Build contracts (no LLM call)
        tasks = []
        for i, (agent_id, task_desc) in enumerate(routes):
            task_id = f"t{i+1:03d}"
            contract = self._build_contract(task_id, agent_id, task_desc)
            self.orch_state.register_task(task_id, agent_id)
            tasks.append(contract)

        # 3. Execute
        plan = TaskPlanLite(goal, tasks)

        if len(tasks) == 1:
            result = self._run_single(tasks[0])
            plan.results[tasks[0].task_id] = result
        else:
            results = self._run_parallel(tasks)
            plan.results.update(results)

        # 4. Aggregate
        final = self._aggregate(goal, plan)

        # 5. Batched learning
        self._tasks_since_learning += 1
        if (self.enable_learning and
            self._tasks_since_learning >= self.learning_batch_size):
            self._run_batched_learning()
            self._tasks_since_learning = 0

        successes = sum(1 for r in plan.results.values() if r.success)
        print(f"\n[OrchestratorLite] Done — {successes}/{len(plan.results)} tasks succeeded")
        return final

    # ── Contract building ─────────────────────────────────────────────────────

    def _build_contract(self, task_id: str, agent_id: str, description: str) -> ExecutionContract:
        tool_sets = {
            "agent_research": ["web_search", "http_get"],
            "agent_code":     ["run_python", "write_file", "read_file"],
            "agent_writer":   ["read_file", "write_file"],
            "agent_file":     ["read_file", "write_file", "list_files"],
            "agent_data":     ["run_python", "read_file"],
            "agent_summary":  ["read_file"],
        }
        output_keys = {
            "agent_research": ["summary", "sources"],
            "agent_code":     ["code", "explanation"],
            "agent_writer":   ["title", "content"],
            "agent_file":     ["files_modified", "summary"],
            "agent_data":     ["analysis", "insights"],
            "agent_summary":  ["summary", "key_points"],
        }

        return ExecutionContract(
            task_id=task_id,
            agent_id=agent_id,
            description=description,
            max_tokens=3000,
            max_tool_calls=3,        # hard cap: 3 tool rounds
            allowed_tools=tool_sets.get(agent_id, ["read_file"]),
            required_output_keys=output_keys.get(agent_id, ["summary"]),
            completion_conditions=["no_error_flag"],
            output_path=f"{self.workspace}/{agent_id}/result_{task_id}.json",
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run_single(self, contract: ExecutionContract) -> ContractResult:
        agent = self._agents.get(contract.agent_id)
        if not agent:
            return ContractResult(contract.task_id, contract.agent_id, False, {},
                                  error=f"Unknown agent: {contract.agent_id}")
        return agent.run(contract)

    def _run_parallel(self, contracts: list[ExecutionContract]) -> dict[str, ContractResult]:
        results = {}
        with ThreadPoolExecutor(max_workers=len(contracts)) as pool:
            futures = {pool.submit(self._run_single, c): c.task_id for c in contracts}
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    results[tid] = future.result()
                except Exception as e:
                    contract = next(c for c in contracts if c.task_id == tid)
                    results[tid] = ContractResult(tid, contract.agent_id, False, {}, error=str(e))
        return results

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, goal: str, plan: TaskPlanLite) -> dict:
        outputs = {}
        for tid, result in plan.results.items():
            outputs[tid] = {
                "agent": plan.tasks[tid].agent_id if tid in plan.tasks else "?",
                "success": result.success,
                "output": result.output,
                "tokens": result.tokens_used,
            }

        agg_path = Path(self.workspace) / "_final_output.json"
        agg_data = {"goal": goal, "tasks": outputs, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        agg_path.write_text(json.dumps(agg_data, indent=2))

        successes = sum(1 for r in plan.results.values() if r.success)
        return {
            "goal": goal,
            "success": successes == len(plan.results),
            "tasks_succeeded": successes,
            "tasks_total": len(plan.results),
            "outputs": outputs,
            "output_file": str(agg_path),
        }

    # ── Batched learning ──────────────────────────────────────────────────────

    def _run_batched_learning(self):
        if not self.cycle:
            return

        print(f"[OrchestratorLite] Running batched learning after {self.learning_batch_size} tasks...")
        for agent_id in self._agents:
            try:
                learner = self.cycle.get_learner(agent_id)
                result = learner.maybe_evolve()
                if result and result.accepted:
                    print(f"[OrchestratorLite] {agent_id} evolved: {len(result.lessons)} lessons")
            except Exception as e:
                print(f"[OrchestratorLite] Learning skip for {agent_id}: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_agents(self) -> dict[str, BaseAgentLite]:
        shared = dict(
            client=self.client,
            registry=self.registry,
            memory=self.memory,
            workspace=self.workspace,
            stream=self.stream,
        )
        return {
            "agent_research": ResearchAgentLite(**shared),
            "agent_code":     CodeAgentLite(**shared),
            "agent_writer":   WriterAgentLite(**shared),
            "agent_file":     FileAgentLite(**shared),
            "agent_data":     DataAgentLite(**shared),
            "agent_summary":  SummaryAgentLite(**shared),
        }


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgentMesh Lite — lean orchestrator")
    parser.add_argument("goal", nargs="?", help="Goal to execute")
    parser.add_argument("--status", action="store_true", help="Show memory stats")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--learning", action="store_true", help="Enable batched learning")
    parser.add_argument("--batch-size", type=int, default=5, help="Tasks between learning cycles")
    args = parser.parse_args()

    orch = OrchestratorLite(
        enable_learning=args.learning,
        learning_batch_size=args.batch_size,
    )

    if args.status:
        stats = orch.memory.stats()
        print(json.dumps(stats, indent=2))
    elif args.test:
        result = orch.run("What is AgentMesh?")
        print(json.dumps(result, indent=2))
    elif args.goal:
        result = orch.run(args.goal)
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()
