"""
AgentMesh · core/skill_registry.py
Shared Skill / Tool Registry — all agents load tools from here on startup.

Design principles from the research:
  - Agents only see tools they are permitted to use (from their contract)
  - Vercel removed 80% of tools and got better results → keep sets SMALL
  - New skills = drop a JSON file in /skills/ → auto-registered
  - Skills directory layout:
      skills/
      ├── web_search.json
      ├── read_file.json
      ├── write_file.json
      ├── run_python.json
      └── ...
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field


# ── Skill definition ──────────────────────────────────────────────────────────

@dataclass
class Skill:
    name: str
    description: str
    parameters: dict        # JSON-schema properties block
    required: list[str]     # required parameter names
    handler: Optional[Callable] = field(default=None, repr=False)
    tags: list[str] = field(default_factory=list)   # e.g. ["read", "io"]
    safe: bool = True       # False = needs extra orchestrator approval

    def to_openai_tool(self) -> dict:
        """Format for Ollama / OpenAI function-calling API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_json(self) -> dict:
        """Serialisable form (no handler)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "tags": self.tags,
            "safe": self.safe,
        }


# ── Registry ──────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Central store of all callable tools.
    Agents query this for their permitted tool list on startup.
    """

    def __init__(self, skills_dir: str = "skills"):
        self._skills: dict[str, Skill] = {}
        self._handlers: dict[str, Callable] = {}
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # Register built-in skills
        self._register_builtins()

        # Auto-discover JSON skill definitions from skills/
        self._load_from_disk()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        if skill.handler:
            self._handlers[skill.name] = skill.handler

    def register_handler(self, name: str, fn: Callable) -> None:
        """Attach a Python function to an existing skill definition."""
        self._handlers[name] = fn
        if name in self._skills:
            self._skills[name].handler = fn

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def get_for_agent(self, allowed_tools: list[str]) -> list[Skill]:
        """Return only the tools this agent's contract permits."""
        return [self._skills[t] for t in allowed_tools if t in self._skills]

    def get_openai_tools(self, allowed_tools: list[str]) -> list[dict]:
        """Formatted for Ollama function-calling API."""
        return [s.to_openai_tool() for s in self.get_for_agent(allowed_tools)]

    def list_all(self) -> list[str]:
        return sorted(self._skills.keys())

    def list_by_tag(self, tag: str) -> list[str]:
        return [n for n, s in self._skills.items() if tag in s.tags]

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, name: str, arguments: dict) -> Any:
        """Call a skill's handler. Returns result or raises SkillError."""
        if name not in self._skills:
            raise SkillError(f"Unknown skill: {name}")
        handler = self._handlers.get(name)
        if handler is None:
            raise SkillError(f"Skill '{name}' has no handler registered")
        try:
            return handler(**arguments)
        except Exception as e:
            raise SkillError(f"Skill '{name}' execution failed: {e}") from e

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_disk(self, name: str) -> Path:
        """Export a skill definition to skills/{name}.json"""
        skill = self._skills.get(name)
        if not skill:
            raise KeyError(name)
        path = self.skills_dir / f"{name}.json"
        path.write_text(json.dumps(skill.to_json(), indent=2))
        return path

    def _load_from_disk(self) -> None:
        """Auto-discover .json files in skills/ directory."""
        for path in self.skills_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                skill = Skill(
                    name=data["name"],
                    description=data["description"],
                    parameters=data.get("parameters", {}),
                    required=data.get("required", []),
                    tags=data.get("tags", []),
                    safe=data.get("safe", True),
                )
                # Don't overwrite a built-in that already has a handler
                if skill.name not in self._skills:
                    self._skills[skill.name] = skill
            except Exception as e:
                print(f"[SkillRegistry] Failed to load {path.name}: {e}")

    # ── Built-in skills ───────────────────────────────────────────────────────

    def _register_builtins(self) -> None:
        builtins = [
            Skill(
                name="read_file",
                description="Read the contents of a file at a given path.",
                parameters={
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                },
                required=["path"],
                tags=["io", "read"],
                handler=self._handle_read_file,
            ),
            Skill(
                name="write_file",
                description="Write text content to a file, creating it if it does not exist.",
                parameters={
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Text content"},
                    "append": {"type": "boolean", "description": "Append instead of overwrite", "default": False},
                },
                required=["path", "content"],
                tags=["io", "write"],
                handler=self._handle_write_file,
            ),
            Skill(
                name="list_files",
                description="List files in a directory.",
                parameters={
                    "directory": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "description": "Glob pattern e.g. *.py", "default": "*"},
                },
                required=["directory"],
                tags=["io", "read"],
                handler=self._handle_list_files,
            ),
            Skill(
                name="run_python",
                description="Execute a Python code snippet and return stdout.",
                parameters={
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                required=["code"],
                tags=["exec"],
                safe=False,
                handler=self._handle_run_python,
            ),
            Skill(
                name="web_search",
                description="Search the web for information on a query. Returns top results as text.",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return", "default": 5},
                },
                required=["query"],
                tags=["web", "read"],
                handler=self._handle_web_search,
            ),
            Skill(
                name="http_get",
                description="Fetch the content of a URL via HTTP GET.",
                parameters={
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                required=["url"],
                tags=["web", "read"],
                handler=self._handle_http_get,
            ),
            Skill(
                name="summarize_text",
                description="Summarize a given text into key points. Returns a condensed version.",
                parameters={
                    "text": {"type": "string", "description": "Text to summarize"},
                    "max_sentences": {"type": "integer", "description": "Max sentences in summary", "default": 5},
                },
                required=["text"],
                tags=["text", "processing"],
                handler=self._handle_summarize_text,
            ),
            Skill(
                name="analyze_csv",
                description="Load and analyze a CSV file. Returns column stats, row count, and data preview.",
                parameters={
                    "path": {"type": "string", "description": "Path to CSV file"},
                    "max_rows": {"type": "integer", "description": "Max rows to preview", "default": 10},
                },
                required=["path"],
                tags=["data", "io"],
                handler=self._handle_analyze_csv,
            ),
            Skill(
                name="generate_chart",
                description="Generate a chart from data and save as PNG. Uses matplotlib.",
                parameters={
                    "data": {"type": "object", "description": "Chart data as {labels: [], values: []}"},
                    "chart_type": {"type": "string", "description": "Chart type: bar, line, pie", "default": "bar"},
                    "output_path": {"type": "string", "description": "Where to save the PNG"},
                },
                required=["data", "output_path"],
                tags=["data", "visualization"],
                handler=self._handle_generate_chart,
            ),
        ]
        for s in builtins:
            self.register(s)

    # ── Built-in handlers ─────────────────────────────────────────────────────

    @staticmethod
    def _handle_read_file(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        return {"content": p.read_text(), "size_bytes": p.stat().st_size}

    @staticmethod
    def _handle_write_file(path: str, content: str, append: bool = False) -> dict:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode) as f:
            f.write(content)
        return {"written": True, "path": str(p), "bytes": len(content)}

    @staticmethod
    def _handle_list_files(directory: str, pattern: str = "*") -> dict:
        d = Path(directory)
        if not d.exists():
            return {"error": f"Directory not found: {directory}"}
        files = [str(f) for f in d.glob(pattern)]
        return {"files": files, "count": len(files)}

    @staticmethod
    def _handle_run_python(code: str, timeout: int = 30) -> dict:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    @staticmethod
    def _handle_web_search(query: str, max_results: int = 5) -> dict:
        """
        Real web search — tries multiple free backends in order:
        1. SearXNG (searx.be) — JSON API, no key needed
        2. DuckDuckGo instant answer API
        3. Mojeek (independent index)
        """
        import urllib.request, urllib.parse, json as _json, re, html as _html

        q = urllib.parse.quote_plus(query)

        # ── Backend 1: SearXNG JSON ───────────────────────────────────────
        searx_instances = [
            "https://searx.be/search",
            "https://search.bus-hit.me/search",
            "https://searxng.site/search",
        ]
        for base in searx_instances:
            try:
                url = f"{base}?q={q}&format=json&language=en"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = _json.loads(r.read())
                results = []
                for item in data.get("results", [])[:max_results]:
                    results.append({
                        "title":   item.get("title", ""),
                        "snippet": item.get("content", ""),
                        "url":     item.get("url", ""),
                    })
                if results:
                    return {"results": results, "query": query,
                            "count": len(results), "source": base}
            except Exception:
                continue

        # ── Backend 2: DDG Lite (updated selectors) ───────────────────────
        try:
            url = f"https://lite.duckduckgo.com/lite/?q={q}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
            })
            with urllib.request.urlopen(req, timeout=8) as r:
                body = r.read().decode("utf-8", errors="replace")
            results = []
            # DDG Lite uses <a class="result-link"> and <td class="result-snippet">
            links    = re.findall(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, re.S)
            snippets = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', body, re.S)
            for i, (href, title) in enumerate(links[:max_results]):
                results.append({
                    "title":   _html.unescape(re.sub(r"<[^>]+>","",title)).strip(),
                    "snippet": _html.unescape(re.sub(r"<[^>]+>","",snippets[i])).strip() if i < len(snippets) else "",
                    "url":     href,
                })
            if results:
                return {"results": results, "query": query,
                        "count": len(results), "source": "ddg-lite"}
        except Exception:
            pass

        # ── Backend 3: DuckDuckGo Instant Answer (summaries only) ────────
        try:
            url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(url, headers={"User-Agent": "AgentMesh/1.0"})
            with urllib.request.urlopen(req, timeout=6) as r:
                data = _json.loads(r.read())
            results = []
            if data.get("AbstractText"):
                results.append({
                    "title":   data.get("Heading",""),
                    "snippet": data["AbstractText"],
                    "url":     data.get("AbstractURL",""),
                })
            for topic in data.get("RelatedTopics",[])[:max_results-1]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title":   topic.get("Text","")[:80],
                        "snippet": topic.get("Text",""),
                        "url":     topic.get("FirstURL",""),
                    })
            if results:
                return {"results": results, "query": query,
                        "count": len(results), "source": "ddg-instant"}
        except Exception:
            pass

        return {"results": [], "error": "All search backends failed — check network"}

    @staticmethod
    def _handle_http_get(url: str) -> dict:
        import urllib.request, re, html as _html
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
            })
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read(80_000).decode("utf-8", errors="replace")

            # Strip scripts, styles, nav boilerplate
            raw = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.S|re.I)
            raw = re.sub(r"<style[^>]*>.*?</style>",  "", raw, flags=re.S|re.I)
            raw = re.sub(r"<nav[^>]*>.*?</nav>",      "", raw, flags=re.S|re.I)
            raw = re.sub(r"<footer[^>]*>.*?</footer>","", raw, flags=re.S|re.I)
            raw = re.sub(r"<header[^>]*>.*?</header>","", raw, flags=re.S|re.I)

            # Convert to plain text
            text = re.sub(r"<[^>]+>", " ", raw)
            text = _html.unescape(text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            # Hard cap: 3000 chars — enough for the agent to read, not enough to loop
            if len(text) > 3000:
                text = text[:3000] + "\n...[truncated]"

            return {"content": text, "url": url, "chars": len(text)}
        except Exception as e:
            return {"error": str(e), "url": url}

    @staticmethod
    def _handle_summarize_text(text: str, max_sentences: int = 5) -> dict:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= max_sentences:
            return {"summary": text, "sentence_count": len(sentences)}
        summary = " ".join(sentences[:max_sentences])
        return {"summary": summary, "original_sentences": len(sentences), "summary_sentences": max_sentences}

    @staticmethod
    def _handle_analyze_csv(path: str, max_rows: int = 10) -> dict:
        import csv
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(dict(row))
            # Re-read for full stats
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
            columns = list(all_rows[0].keys()) if all_rows else []
            col_stats = {}
            for col in columns:
                values = [r.get(col, "") for r in all_rows]
                numeric = []
                for v in values:
                    try:
                        numeric.append(float(v))
                    except (ValueError, TypeError):
                        pass
                if numeric:
                    col_stats[col] = {
                        "type": "numeric",
                        "count": len(numeric),
                        "min": min(numeric),
                        "max": max(numeric),
                        "mean": round(sum(numeric) / len(numeric), 2),
                    }
                else:
                    col_stats[col] = {"type": "text", "unique": len(set(values)), "count": len(values)}
            return {"columns": columns, "row_count": len(all_rows), "preview": rows, "column_stats": col_stats}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _handle_generate_chart(data: dict, chart_type: str = "bar", output_path: str = "chart.png") -> dict:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {"error": "matplotlib not installed. Run: pip install matplotlib"}
        labels = data.get("labels", [])
        values = data.get("values", [])
        if not labels or not values:
            return {"error": "data must contain 'labels' and 'values' lists"}
        fig, ax = plt.subplots(figsize=(10, 6))
        if chart_type == "pie":
            ax.pie(values, labels=labels, autopct="%1.1f%%")
        elif chart_type == "line":
            ax.plot(labels, values, marker="o")
            ax.tick_params(axis="x", rotation=45)
        else:  # bar
            ax.bar(labels, values)
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(data.get("title", "Chart"))
        fig.tight_layout()
        from pathlib import Path as _Path
        _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return {"saved": output_path, "chart_type": chart_type, "data_points": len(labels)}


class SkillError(Exception):
    pass
