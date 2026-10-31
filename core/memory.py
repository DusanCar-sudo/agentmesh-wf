"""
AgentMesh · core/memory.py
Agent Persistence — memory that survives across sessions.

Four memory layers (mirrors human cognitive architecture):
  EPISODIC   — raw task history, verbatim traces. Never summarised.
               Stanford finding: summaries drop accuracy 50%→34.9%. Keep raw.
  SEMANTIC   — what the agent has learned (lessons, rules, patterns)
  PROCEDURAL — skill effectiveness scores (which tools work for which tasks)
  WORKING    — current session scratchpad (in FileBackedState)

All backed by SQLite — zero extra services, works on Fedora out of the box.
"""

from __future__ import annotations
import json
import math
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


DB_PATH = "workspace/memory.db"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """A completed task — the raw record of what happened."""
    episode_id: str
    agent_id: str
    task_id: str
    task_description: str
    success: bool
    tokens_used: int
    tool_calls: int
    elapsed_seconds: float
    output_summary: str          # brief human-readable summary
    lessons: list[str]           # extracted lessons from this episode
    traces: list[dict]           # verbatim trace events (never summarised)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    tags: list[str] = field(default_factory=list)
    # Server-compatible fields
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    workflow_nodes: list[str] = field(default_factory=list)
    input_text: str = ""
    rating: Optional[int] = None


@dataclass
class Lesson:
    """A specific learned rule or pattern."""
    lesson_id: str
    agent_id: str
    content: str                 # e.g. "Always verify file exists before reading"
    source_episode: str          # which episode generated this
    confidence: float = 1.0     # 0.0–1.0, decays if lesson leads to failures
    applies_to: list[str] = field(default_factory=list)  # task type tags
    reinforcements: int = 1      # how many times this lesson was confirmed
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class SkillStat:
    """Tracks effectiveness of a tool/skill across all agents."""
    skill_name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    avg_tokens: float = 0.0
    last_used: str = ""

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successes / self.total_calls

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate


# ── Memory store ──────────────────────────────────────────────────────────────

class AgentMemory:
    """
    Persistent memory for all agents in the mesh.
    Single SQLite database — all agents share the same store,
    but query by agent_id to get their own view.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Episodic memory ───────────────────────────────────────────────────────

    def store_episode(self, episode: Episode) -> str:
        """
        Save a completed task. Called by every agent after task completion.
        Stores verbatim traces — research shows raw > summarised every time.
        """
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO episodes
                (episode_id, agent_id, task_id, task_description,
                 success, tokens_used, tool_calls, elapsed_seconds,
                 output_summary, lessons_json, traces_json, tags_json, timestamp,
                 provider, model, latency_ms, workflow_nodes_json, input_text, rating)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                episode.episode_id,
                episode.agent_id,
                episode.task_id,
                episode.task_description,
                int(episode.success),
                episode.tokens_used,
                episode.tool_calls,
                episode.elapsed_seconds,
                episode.output_summary,
                json.dumps(episode.lessons),
                json.dumps(episode.traces),
                json.dumps(episode.tags),
                episode.timestamp,
                episode.provider,
                episode.model,
                episode.latency_ms,
                json.dumps(episode.workflow_nodes),
                episode.input_text,
                episode.rating,
            ))
        return episode.episode_id

    def recall_similar(
        self,
        query: str,
        agent_id: str = None,
        n: int = 5,
        success_only: bool = False,
    ) -> list[Episode]:
        """
        Find past episodes similar to the current task description.
        Uses lightweight TF-IDF-style word overlap (no embeddings needed).
        For a full project, swap with ChromaDB vector search.
        """
        with self._conn() as c:
            sql = "SELECT * FROM episodes"
            params = []
            conditions = []
            if agent_id:
                conditions.append("agent_id = ?")
                params.append(agent_id)
            if success_only:
                conditions.append("success = 1")
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY timestamp DESC LIMIT 200"
            rows = c.execute(sql, params).fetchall()

        episodes = [self._row_to_episode(r) for r in rows]

        # Rank by word overlap with query
        query_words = set(query.lower().split())
        scored = []
        for ep in episodes:
            ep_words = set(ep.task_description.lower().split())
            overlap = len(query_words & ep_words)
            if overlap > 0:
                scored.append((overlap, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:n]]

    def get_failures(self, agent_id: str = None, limit: int = 20) -> list[Episode]:
        """Return recent failed episodes — primary input for self-evolution."""
        with self._conn() as c:
            if agent_id:
                rows = c.execute(
                    "SELECT * FROM episodes WHERE success=0 AND agent_id=? "
                    "ORDER BY timestamp DESC LIMIT ?", (agent_id, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM episodes WHERE success=0 "
                    "ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_recent(self, agent_id: str = None, limit: int = 10) -> list[Episode]:
        with self._conn() as c:
            if agent_id:
                rows = c.execute(
                    "SELECT * FROM episodes WHERE agent_id=? "
                    "ORDER BY timestamp DESC LIMIT ?", (agent_id, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def episode_count(self, agent_id: str = None) -> int:
        with self._conn() as c:
            if agent_id:
                return c.execute(
                    "SELECT COUNT(*) FROM episodes WHERE agent_id=?", (agent_id,)
                ).fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    # ── Semantic memory (lessons) ─────────────────────────────────────────────

    def store_lesson(self, lesson: Lesson) -> str:
        """Save a learned rule. Agent will prepend these to future system prompts."""
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO lessons
                (lesson_id, agent_id, content, source_episode,
                 confidence, applies_to_json, reinforcements, timestamp)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                lesson.lesson_id,
                lesson.agent_id,
                lesson.content,
                lesson.source_episode,
                lesson.confidence,
                json.dumps(lesson.applies_to),
                lesson.reinforcements,
                lesson.timestamp,
            ))
        return lesson.lesson_id

    def reinforce_lesson(self, lesson_id: str, success: bool) -> None:
        """
        Called when a lesson was applied and the task succeeded/failed.
        Confidence adjusts accordingly — failed lessons decay and get pruned.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT confidence, reinforcements FROM lessons WHERE lesson_id=?",
                (lesson_id,)
            ).fetchone()
            if not row:
                return
            conf, reinforcements = row
            if success:
                new_conf = min(1.0, conf + 0.05)
                new_r = reinforcements + 1
            else:
                new_conf = max(0.0, conf - 0.15)
                new_r = reinforcements
            c.execute(
                "UPDATE lessons SET confidence=?, reinforcements=? WHERE lesson_id=?",
                (new_conf, new_r, lesson_id)
            )
            # Prune lessons with very low confidence
            if new_conf < 0.2:
                c.execute("DELETE FROM lessons WHERE lesson_id=?", (lesson_id,))

    def get_lessons(
        self,
        agent_id: str,
        min_confidence: float = 0.5,
        limit: int = 15,
    ) -> list[Lesson]:
        """
        Retrieve top lessons for an agent.
        Injected into the agent's system prompt before each task.
        """
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM lessons
                WHERE agent_id=? AND confidence >= ?
                ORDER BY confidence DESC, reinforcements DESC
                LIMIT ?
            """, (agent_id, min_confidence, limit)).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def format_lessons_for_prompt(self, agent_id: str) -> str:
        """
        Returns a formatted block to inject into agent system prompts.
        Empty string if no lessons yet.
        """
        lessons = self.get_lessons(agent_id)
        if not lessons:
            return ""
        lines = ["LEARNED RULES (apply these to every task):"]
        for i, l in enumerate(lessons, 1):
            lines.append(f"  {i}. {l.content}  [confidence: {l.confidence:.0%}]")
        return "\n".join(lines)

    # ── Procedural memory (skill stats) ──────────────────────────────────────

    def record_skill_call(
        self,
        skill_name: str,
        success: bool,
        tokens_used: int = 0,
    ) -> None:
        with self._conn() as c:
            row = c.execute(
                "SELECT total_calls, successes, failures, avg_tokens FROM skill_stats WHERE skill_name=?",
                (skill_name,)
            ).fetchone()

            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            if row:
                total, succ, fail, avg_tok = row
                total += 1
                if success:
                    succ += 1
                else:
                    fail += 1
                # running average
                avg_tok = (avg_tok * (total - 1) + tokens_used) / total
                c.execute("""
                    UPDATE skill_stats
                    SET total_calls=?, successes=?, failures=?, avg_tokens=?, last_used=?
                    WHERE skill_name=?
                """, (total, succ, fail, avg_tok, now, skill_name))
            else:
                c.execute("""
                    INSERT INTO skill_stats
                    (skill_name, total_calls, successes, failures, avg_tokens, last_used)
                    VALUES (?,1,?,?,?,?)
                """, (
                    skill_name,
                    1 if success else 0,
                    0 if success else 1,
                    float(tokens_used),
                    now,
                ))

    def get_skill_stats(self) -> list[SkillStat]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM skill_stats ORDER BY total_calls DESC").fetchall()
        return [SkillStat(
            skill_name=r[0],
            total_calls=r[1],
            successes=r[2],
            failures=r[3],
            avg_tokens=r[4],
            last_used=r[5],
        ) for r in rows]

    def get_weak_skills(self, threshold: float = 0.5) -> list[SkillStat]:
        """Skills with success rate below threshold — candidates for removal."""
        return [s for s in self.get_skill_stats()
                if s.total_calls >= 5 and s.success_rate < threshold]

    # ── Cross-session context builder ─────────────────────────────────────────

    def build_context_for_task(self, agent_id: str, task_description: str) -> str:
        """
        Builds a rich context block to inject before a new task.
        Combines: similar past episodes + current lessons + skill warnings.
        This is what makes the agent smarter over time.
        """
        blocks = []

        # 1. Relevant past experience
        similar = self.recall_similar(task_description, agent_id=agent_id, n=3)
        if similar:
            blocks.append("RELEVANT PAST EXPERIENCE:")
            for ep in similar:
                status = "SUCCESS" if ep.success else "FAILED"
                blocks.append(f"  [{status}] {ep.task_description}")
                if ep.lessons:
                    for lesson in ep.lessons[:2]:
                        blocks.append(f"    → {lesson}")

        # 2. Accumulated lessons
        lesson_block = self.format_lessons_for_prompt(agent_id)
        if lesson_block:
            blocks.append(lesson_block)

        # 3. Skill warnings
        weak = self.get_weak_skills()
        if weak:
            weak_names = [s.skill_name for s in weak]
            blocks.append(f"LOW-RELIABILITY TOOLS (use with caution): {', '.join(weak_names)}")

        return "\n\n".join(blocks)

    # ── Stats / introspection ─────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            success = c.execute("SELECT COUNT(*) FROM episodes WHERE success=1").fetchone()[0]
            lessons = c.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            skills = c.execute("SELECT COUNT(*) FROM skill_stats").fetchone()[0]
        return {
            "total_episodes": total,
            "success_rate": round(success / total, 3) if total else 0,
            "total_lessons": lessons,
            "tracked_skills": skills,
        }

    def get_provider_stats(self) -> list[dict]:
        """Compute provider statistics from episodes."""
        with self._conn() as c:
            rows = c.execute("""
                SELECT provider, model,
                       COUNT(*) as runs,
                       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successes,
                       AVG(latency_ms) as avg_latency,
                       AVG(tokens_used) as avg_tokens
                FROM episodes WHERE provider != ''
                GROUP BY provider, model
                ORDER BY runs DESC
            """).fetchall()
        return [
            {
                "provider": r["provider"],
                "model": r["model"],
                "runs": r["runs"],
                "successes": r["successes"],
                "success_rate": round(r["successes"] / r["runs"], 2) if r["runs"] else 0,
                "avg_latency_ms": round(r["avg_latency"] or 0, 1),
                "avg_tokens": round(r["avg_tokens"] or 0),
            }
            for r in rows
        ]

    def get_workflow_stats(self) -> list[dict]:
        """Compute workflow pattern stats from episodes."""
        with self._conn() as c:
            rows = c.execute("""
                SELECT workflow_nodes_json,
                       COUNT(*) as runs,
                       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successes
                FROM episodes WHERE workflow_nodes_json != '[]' AND workflow_nodes_json != ''
                GROUP BY workflow_nodes_json
                ORDER BY runs DESC
                LIMIT 10
            """).fetchall()
        result = []
        for r in rows:
            nodes = json.loads(r["workflow_nodes_json"])
            pattern = " -> ".join(nodes)
            result.append({
                "pattern": pattern,
                "runs": r["runs"],
                "successes": r["successes"],
                "success_rate": round(r["successes"] / r["runs"], 2) if r["runs"] else 0,
            })
        return result

    def get_lessons_flat(self) -> list[dict]:
        """Return lessons in flat format for server compatibility."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT lesson_id, content, confidence, reinforcements, timestamp FROM lessons "
                "ORDER BY confidence DESC LIMIT 50"
            ).fetchall()
        return [
            {"key": r["lesson_id"], "text": r["content"], "confidence": r["confidence"],
             "reinforcements": r["reinforcements"], "updated": r["timestamp"]}
            for r in rows
        ]

    def rate_episode(self, episode_id: str, rating: int) -> bool:
        """Rate an episode 1-5. Returns True if found."""
        with self._conn() as c:
            c.execute("UPDATE episodes SET rating=? WHERE episode_id=?",
                      (max(1, min(5, rating)), episode_id))
            return c.total_changes > 0

    # ── SQLite helpers ────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate_columns(self, cursor) -> None:
        """Add new columns to existing tables if missing."""
        existing = {row[1] for row in cursor.execute("PRAGMA table_info(episodes)").fetchall()}
        migrations = {
            "provider": "ALTER TABLE episodes ADD COLUMN provider TEXT DEFAULT ''",
            "model": "ALTER TABLE episodes ADD COLUMN model TEXT DEFAULT ''",
            "latency_ms": "ALTER TABLE episodes ADD COLUMN latency_ms REAL DEFAULT 0.0",
            "workflow_nodes_json": "ALTER TABLE episodes ADD COLUMN workflow_nodes_json TEXT DEFAULT '[]'",
            "input_text": "ALTER TABLE episodes ADD COLUMN input_text TEXT DEFAULT ''",
            "rating": "ALTER TABLE episodes ADD COLUMN rating INTEGER",
        }
        for col, sql in migrations.items():
            if col not in existing:
                cursor.execute(sql)
        # Ensure index exists
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ep_provider ON episodes(provider)")

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id      TEXT PRIMARY KEY,
                    agent_id        TEXT NOT NULL,
                    task_id         TEXT,
                    task_description TEXT,
                    success         INTEGER,
                    tokens_used     INTEGER,
                    tool_calls      INTEGER,
                    elapsed_seconds REAL,
                    output_summary  TEXT,
                    lessons_json    TEXT,
                    traces_json     TEXT,
                    tags_json       TEXT,
                    timestamp       TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_ep_agent ON episodes(agent_id);
                CREATE INDEX IF NOT EXISTS idx_ep_success ON episodes(success);

                CREATE TABLE IF NOT EXISTS lessons (
                    lesson_id       TEXT PRIMARY KEY,
                    agent_id        TEXT NOT NULL,
                    content         TEXT,
                    source_episode  TEXT,
                    confidence      REAL,
                    applies_to_json TEXT,
                    reinforcements  INTEGER,
                    timestamp       TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_lesson_agent ON lessons(agent_id);

                CREATE TABLE IF NOT EXISTS skill_stats (
                    skill_name  TEXT PRIMARY KEY,
                    total_calls INTEGER,
                    successes   INTEGER,
                    failures    INTEGER,
                    avg_tokens  REAL,
                    last_used   TEXT
                );
            """)
            # Migration: add new columns + indexes to existing tables
            self._migrate_columns(c)

    def _row_to_episode(self, row) -> Episode:
        return Episode(
            episode_id=row["episode_id"],
            agent_id=row["agent_id"],
            task_id=row["task_id"] or "",
            task_description=row["task_description"] or "",
            success=bool(row["success"]),
            tokens_used=row["tokens_used"] or 0,
            tool_calls=row["tool_calls"] or 0,
            elapsed_seconds=row["elapsed_seconds"] or 0.0,
            output_summary=row["output_summary"] or "",
            lessons=json.loads(row["lessons_json"] or "[]"),
            traces=json.loads(row["traces_json"] or "[]"),
            tags=json.loads(row["tags_json"] or "[]"),
            timestamp=row["timestamp"] or "",
            provider=row["provider"] or "",
            model=row["model"] or "",
            latency_ms=row["latency_ms"] or 0.0,
            workflow_nodes=json.loads(row["workflow_nodes_json"] or "[]") if "workflow_nodes_json" in row.keys() else [],
            input_text=row["input_text"] or "" if "input_text" in row.keys() else "",
            rating=row["rating"] if "rating" in row.keys() else None,
        )

    def _row_to_lesson(self, row) -> Lesson:
        return Lesson(
            lesson_id=row["lesson_id"],
            agent_id=row["agent_id"],
            content=row["content"],
            source_episode=row["source_episode"],
            confidence=row["confidence"],
            applies_to=json.loads(row["applies_to_json"] or "[]"),
            reinforcements=row["reinforcements"],
            timestamp=row["timestamp"],
        )


# ── Convenience factory ───────────────────────────────────────────────────────

def make_episode_id(agent_id: str, task_id: str) -> str:
    raw = f"{agent_id}:{task_id}:{time.time()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def make_lesson_id(agent_id: str, content: str) -> str:
    raw = f"{agent_id}:{content}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
