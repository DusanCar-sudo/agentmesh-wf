"""
AgentMesh · core/learner.py
Recursive Learning — the self-evolution loop.

From the research:
  Tsinghua 2026: self-evolution was the ONLY module that CONSISTENTLY helped.
    +4.8 on SWE-bench, +2.7 on OS-World.
    Mechanism: narrow attempt loop, broaden only when failure signals justify it.

  Stanford meta-harness: the orchestrator reads raw failure traces and rewrites
    the harness. 10M tokens per iteration. Accuracy: 50% vs 34.6% without traces.

This module implements both:
  1. RecursiveLearner  — per-agent self-evolution (learns from own failures)
  2. HarnessOptimizer  — orchestrator-level evolution (rewrites decomposition strategy)

Both use DeepSeek-R1 (reasoner) for the diagnosis pass — cheap, strong at structured
reasoning, and the task (pattern extraction from traces) is exactly what it's built for.
"""

from __future__ import annotations
import fcntl
import json
import time
import hashlib
from pathlib import Path
from typing import Optional

from .memory import AgentMemory, Episode, Lesson, make_lesson_id
from .hermes_client import HermesClient, Provider
from .utils import parse_json_from_llm


# ── Evolution result ──────────────────────────────────────────────────────────

class EvolutionResult:
    def __init__(
        self,
        accepted: bool,
        lessons: list[str],
        prompt_addendum: str,
        diagnosis: str,
        iteration: int,
    ):
        self.accepted = accepted
        self.lessons = lessons               # concrete rules extracted
        self.prompt_addendum = prompt_addendum  # text injected into system prompt
        self.diagnosis = diagnosis           # why evolution was triggered
        self.iteration = iteration
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def __repr__(self):
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return (f"EvolutionResult({status}, iter={self.iteration}, "
                f"lessons={len(self.lessons)})")


# ── Per-agent recursive learner ───────────────────────────────────────────────

class RecursiveLearner:
    """
    Reads an agent's failure traces, extracts lessons, evolves system prompt.

    The acceptance gate is critical — only narrow the attempt loop when you
    have clear signal. Broadening is expensive and the research shows it hurts.
    """

    def __init__(
        self,
        agent_id: str,
        memory: AgentMemory,
        client: HermesClient,
        min_failures_to_trigger: int = 2,
        workspace: str = "workspace",
    ):
        self.agent_id = agent_id
        self.memory = memory
        self.client = client
        self.min_failures = min_failures_to_trigger
        self.workspace = workspace
        self._iteration = self._load_iteration()

    def maybe_evolve(self, force: bool = False) -> Optional[EvolutionResult]:
        """
        Check if evolution is warranted. Call after each completed task.
        Returns EvolutionResult if evolution ran, None if not triggered.

        Trigger condition: >= min_failures recent failures with shared patterns.
        """
        failures = self.memory.get_failures(self.agent_id, limit=10)
        if not force and len(failures) < self.min_failures:
            return None

        # Only evolve if failure rate in recent history is high enough
        recent = self.memory.get_recent(self.agent_id, limit=10)
        if not force and recent:
            fail_rate = sum(1 for e in recent if not e.success) / len(recent)
            if fail_rate < 0.3:
                return None  # Less than 30% failure — don't evolve yet

        return self._run_evolution(failures)

    def _run_evolution(self, failures: list[Episode]) -> EvolutionResult:
        """
        Core evolution loop:
        1. Diagnose failure patterns using DeepSeek-R1
        2. Extract concrete lessons
        3. Run acceptance gate
        4. If accepted: store lessons + update prompt addendum
        """
        self._iteration += 1
        self._save_iteration()

        # Build diagnosis prompt from raw traces (never summarised)
        trace_dump = self._format_failures_for_prompt(failures)
        existing_lessons = self.memory.format_lessons_for_prompt(self.agent_id)

        diagnosis_prompt = f"""You are analysing failure patterns for AI agent "{self.agent_id}".

RECENT FAILURES (raw traces — {len(failures)} episodes):
{trace_dump}

EXISTING LESSONS ALREADY APPLIED:
{existing_lessons or "(none yet)"}

Your job:
1. Identify the ROOT CAUSE patterns across these failures (not symptoms).
2. Extract 2-5 CONCRETE, ACTIONABLE rules the agent must follow to avoid these failures.
3. Rules must be specific enough to change behaviour — not generic advice.

Format your response as JSON only:
{{
  "diagnosis": "one paragraph explaining the root cause pattern",
  "lessons": [
    "Specific rule 1 — what to do differently",
    "Specific rule 2 — what to check before acting",
    ...
  ],
  "prompt_addendum": "2-3 sentence instruction block to prepend to system prompt"
}}"""

        try:
            response = self.client.complete(
                diagnosis_prompt,
                system="You are a precise AI systems analyst. Respond only with valid JSON.",
                max_tokens=800,
            )
            data = parse_json_from_llm(response)
        except Exception as e:
            data = {
                "diagnosis": f"Evolution failed: {e}",
                "lessons": [],
                "prompt_addendum": "",
            }

        lessons_text = data.get("lessons", [])
        prompt_addendum = data.get("prompt_addendum", "")
        diagnosis = data.get("diagnosis", "")

        # Acceptance gate
        accepted = self._acceptance_gate(lessons_text, failures)

        result = EvolutionResult(
            accepted=accepted,
            lessons=lessons_text,
            prompt_addendum=prompt_addendum,
            diagnosis=diagnosis,
            iteration=self._iteration,
        )

        if accepted:
            self._apply_lessons(lessons_text, failures)
            self._save_prompt_addendum(prompt_addendum)
            print(f"[Learner:{self.agent_id}] Evolution {self._iteration} ACCEPTED — "
                  f"{len(lessons_text)} lessons stored")
        else:
            print(f"[Learner:{self.agent_id}] Evolution {self._iteration} REJECTED — "
                  f"gate did not pass")

        self._log_evolution(result)
        return result

    def _acceptance_gate(self, lessons: list[str], failures: list[Episode]) -> bool:
        """
        Gate: only accept evolution if lessons are non-trivial, specific,
        AND semantically relevant to the actual failures.
        Research finding: discipline narrowing beats expensive broadening.
        Gate is intentionally strict — it's better to skip bad lessons.
        """
        if not lessons:
            return False

        # Reject if lessons are too generic
        generic_phrases = [
            "be more careful", "try harder", "check everything",
            "always verify", "make sure", "be thorough",
        ]
        specific_count = 0
        for lesson in lessons:
            lesson_lower = lesson.lower()
            is_generic = any(phrase in lesson_lower for phrase in generic_phrases)
            if not is_generic and len(lesson) > 30:
                specific_count += 1

        # Need at least 1 specific, actionable lesson
        if specific_count == 0:
            return False

        # Semantic relevance: lessons must reference concepts from the actual failures
        if not self._lessons_are_relevant(lessons, failures):
            print(f"[Learner:{self.agent_id}] REJECTED — lessons not relevant to failures")
            return False

        # Reject if we've had 3+ evolutions recently with no improvement
        recent = self.memory.get_recent(self.agent_id, limit=15)
        if len(recent) >= 10:
            last_10_success = sum(1 for e in recent[-10:] if e.success) / 10
            if last_10_success < 0.2 and self._iteration > 3:
                # More than 3 evolutions and still failing badly
                # Signal to orchestrator: this agent needs task re-scoping, not more lessons
                print(f"[Learner:{self.agent_id}] WARNING: persistent failure despite "
                      f"{self._iteration} evolutions — escalate to orchestrator")
                return False  # Don't keep adding lessons that don't work

        return True

    @staticmethod
    def _lessons_are_relevant(lessons: list[str], failures: list[Episode]) -> bool:
        """
        Check that lessons are semantically relevant to the failure traces.
        Builds a vocabulary from failure descriptions and traces, then verifies
        that each lesson shares meaningful terms with that vocabulary.

        Rejects lessons that are plausible-sounding but unrelated to what
        actually went wrong.
        """
        # Build reference vocabulary from failure episodes
        vocab_words: set[str] = set()
        for ep in failures:
            # Words from task description
            vocab_words.update(ep.task_description.lower().split())
            # Words from trace events
            for trace in ep.traces[-10:]:
                event = trace.get("event", "")
                vocab_words.update(event.lower().split())
                # Include string values from trace dict
                for v in trace.values():
                    if isinstance(v, str):
                        vocab_words.update(v.lower().split())

        # Filter out very short tokens and common stopwords
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "that", "this", "it", "not", "or", "and", "but",
            "if", "then", "so", "no", "yes", "do", "did", "does",
        }
        vocab_words = {w for w in vocab_words if len(w) > 2 and w not in stopwords}

        if not vocab_words:
            # No useful vocabulary — don't block on this check
            return True

        # At least half of lessons must share terms with the failure vocabulary
        relevant_count = 0
        for lesson in lessons:
            lesson_words = set(lesson.lower().split())
            overlap = lesson_words & vocab_words
            if len(overlap) >= 2:
                relevant_count += 1

        # Require majority of lessons to be relevant
        return relevant_count >= max(1, len(lessons) // 2)

    def _apply_lessons(self, lessons: list[str], source_failures: list[Episode]) -> None:
        """Store accepted lessons in memory with deduplication."""
        for content in lessons:
            lesson_id = make_lesson_id(self.agent_id, content)
            # Check if very similar lesson already exists
            existing = self.memory.get_lessons(self.agent_id)
            similar_exists = any(
                self._text_overlap(content, ex.content) > 0.7
                for ex in existing
            )
            if similar_exists:
                continue

            lesson = Lesson(
                lesson_id=lesson_id,
                agent_id=self.agent_id,
                content=content,
                source_episode=source_failures[0].episode_id if source_failures else "",
                confidence=0.75,   # start at 75%, decay if it causes failures
                applies_to=[],
                reinforcements=1,
            )
            self.memory.store_lesson(lesson)

    def get_prompt_addendum(self) -> str:
        """Return the current evolved system prompt addendum."""
        path = Path(f"{self.workspace}/{self.agent_id}/prompt_addendum.txt")
        if path.exists():
            return path.read_text()
        return ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_failures_for_prompt(self, failures: list[Episode]) -> str:
        """
        Format raw failure traces for the diagnosis prompt.
        Verbatim — no summarisation. This is what Stanford found essential.
        """
        parts = []
        for ep in failures[:5]:  # cap at 5 to stay within token budget
            part = [
                f"Episode: {ep.episode_id}",
                f"Task: {ep.task_description}",
                f"Tokens used: {ep.tokens_used} | Tool calls: {ep.tool_calls}",
                "Traces:",
            ]
            for trace in ep.traces[-10:]:  # last 10 trace events per episode
                part.append(f"  [{trace.get('event','')}] {json.dumps({k:v for k,v in trace.items() if k not in ('ts','agent_id')})}")
            parts.append("\n".join(part))
        return "\n\n---\n\n".join(parts)

    def _save_prompt_addendum(self, text: str) -> None:
        path = Path(f"{self.workspace}/{self.agent_id}/prompt_addendum.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(text)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _log_evolution(self, result: EvolutionResult) -> None:
        entry = {
            "ts": result.timestamp,
            "agent_id": self.agent_id,
            "iteration": result.iteration,
            "accepted": result.accepted,
            "lessons_count": len(result.lessons),
            "diagnosis": result.diagnosis[:200],
        }
        log_path = Path(f"{self.workspace}/evolution_log.jsonl")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_iteration(self) -> int:
        path = Path(f"{self.workspace}/{self.agent_id}/_iteration.txt")
        if path.exists():
            try:
                return int(path.read_text().strip())
            except Exception:
                pass
        return 0

    def _save_iteration(self) -> None:
        path = Path(f"{self.workspace}/{self.agent_id}/_iteration.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(str(self._iteration))
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Word-level Jaccard similarity for deduplication."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)


# ── Orchestrator-level harness optimizer ─────────────────────────────────────

class HarnessOptimizer:
    """
    Rewrites the orchestrator's task decomposition strategy based on
    all-agent failure traces. This is the Stanford meta-harness pattern.

    The orchestrator reads what broke across the whole mesh and
    rewrites how it decomposes and delegates tasks.
    """

    def __init__(self, memory: AgentMemory, client: HermesClient, workspace: str = "workspace"):
        self.memory = memory
        self.client = client
        self.workspace = workspace

    def optimize(self, current_goal: str) -> str:
        """
        Analyse all recent failures mesh-wide and rewrite the decomposition strategy.
        Returns updated strategy text the orchestrator uses for planning.
        Called after completing a full multi-agent task cycle.
        """
        # Gather all failures across all agents
        all_failures = self.memory.get_failures(limit=30)
        if len(all_failures) < 3:
            return self.get_current_strategy()

        # Build cross-agent failure analysis
        failure_summary = self._build_cross_agent_summary(all_failures)
        skill_stats = self.memory.get_skill_stats()
        skill_summary = self._format_skill_stats(skill_stats)

        prompt = f"""You are optimising a multi-agent orchestration harness.

CURRENT GOAL TYPE: {current_goal}

CROSS-AGENT FAILURE ANALYSIS ({len(all_failures)} failures):
{failure_summary}

SKILL/TOOL EFFECTIVENESS:
{skill_summary}

CURRENT STRATEGY:
{self.get_current_strategy() or "(no strategy yet — generate initial strategy)"}

Based on these failure patterns, rewrite the orchestration strategy.
Focus on:
1. Which task decomposition patterns fail (and how to fix them)
2. Which agent assignments are ineffective (and better routing)
3. Which tools to avoid or use with caution
4. Token/tool budget adjustments

Respond with JSON only:
{{
  "strategy": "2-4 paragraph strategy text the orchestrator reads before decomposing tasks",
  "routing_rules": [
    "Rule: [task type] → [agent] because [reason]",
    ...
  ],
  "avoid": ["tool or pattern to avoid"],
  "changes": "one sentence summary of what changed vs previous strategy"
}}"""

        try:
            response = self.client.complete(
                prompt,
                system="You are a multi-agent systems architect. Respond only with valid JSON.",
                max_tokens=1000,
            )
            data = parse_json_from_llm(response)
            strategy = data.get("strategy", "")
            routing = data.get("routing_rules", [])
            changes = data.get("changes", "")

            if strategy:
                full_strategy = strategy + "\n\nROUTING RULES:\n" + "\n".join(
                    f"  • {r}" for r in routing
                )
                self._save_strategy(full_strategy, changes)
                print(f"[HarnessOptimizer] Strategy updated: {changes}")
                return full_strategy
        except Exception as e:
            print(f"[HarnessOptimizer] Optimization failed: {e}")

        return self.get_current_strategy()

    def get_current_strategy(self) -> str:
        path = Path(f"{self.workspace}/_orchestrator_strategy.txt")
        if path.exists():
            return path.read_text()
        return ""

    def _build_cross_agent_summary(self, failures: list[Episode]) -> str:
        # Group by agent
        by_agent: dict[str, list[Episode]] = {}
        for ep in failures:
            by_agent.setdefault(ep.agent_id, []).append(ep)

        parts = []
        for agent_id, eps in by_agent.items():
            part = [f"Agent: {agent_id} — {len(eps)} failures"]
            for ep in eps[:3]:
                part.append(f"  Task: {ep.task_description[:80]}")
                if ep.traces:
                    last_trace = ep.traces[-1]
                    part.append(f"  Last event: {last_trace.get('event','')} — "
                                f"{str(last_trace)[:100]}")
            parts.append("\n".join(part))
        return "\n\n".join(parts)

    def _format_skill_stats(self, stats) -> str:
        if not stats:
            return "(no skill data yet)"
        lines = []
        for s in stats[:10]:
            lines.append(f"  {s.skill_name}: {s.success_rate:.0%} success "
                        f"({s.total_calls} calls, avg {s.avg_tokens:.0f} tokens)")
        return "\n".join(lines)

    def _save_strategy(self, strategy: str, changes: str) -> None:
        strategy_path = Path(f"{self.workspace}/_orchestrator_strategy.txt")
        strategy_path.write_text(strategy)
        history_path = Path(f"{self.workspace}/_strategy_history.jsonl")
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "changes": changes,
            "length": len(strategy),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

# ── Learning cycle coordinator ────────────────────────────────────────────────

class LearningCycle:
    """
    Ties it all together. Call after every completed task cycle.

    Flow:
      task completes → store episode → maybe_evolve per agent
                                     → harness_optimizer.optimize (if enough data)
    """

    def __init__(self, memory: AgentMemory, client: HermesClient, workspace: str = "workspace"):
        self.memory = memory
        self.client = client
        self.workspace = workspace
        self._learners: dict[str, RecursiveLearner] = {}

    def get_learner(self, agent_id: str) -> RecursiveLearner:
        if agent_id not in self._learners:
            self._learners[agent_id] = RecursiveLearner(
                agent_id=agent_id,
                memory=self.memory,
                client=self.client,
                workspace=self.workspace,
            )
        return self._learners[agent_id]

    def after_task(
        self,
        agent_id: str,
        episode: Episode,
        evolve: bool = True,
    ) -> Optional[EvolutionResult]:
        """
        Call this after every agent task completes.
        1. Stores the episode
        2. Optionally triggers self-evolution
        """
        self.memory.store_episode(episode)

        if evolve:
            learner = self.get_learner(agent_id)
            return learner.maybe_evolve()
        return None

    def get_enriched_system_prompt(
        self,
        agent_id: str,
        base_system_prompt: str,
    ) -> str:
        """
        Returns the base system prompt enriched with:
          - Accumulated lessons (from memory)
          - Evolved prompt addendum (from learner)
        Call this before every task to give agents their accumulated knowledge.
        """
        learner = self.get_learner(agent_id)

        parts = [base_system_prompt]

        # Inject cross-session context (similar past tasks + lessons)
        # Will be empty on first task — builds up over time
        context = self.memory.format_lessons_for_prompt(agent_id)
        if context:
            parts.append(context)

        addendum = learner.get_prompt_addendum()
        if addendum:
            parts.append(f"EVOLVED BEHAVIOUR (from past learning):\n{addendum}")

        return "\n\n".join(parts)
