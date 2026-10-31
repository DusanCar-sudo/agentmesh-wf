#!/usr/bin/env python3
"""
AgentMesh · research_cascade.py
Multi-pass deep research using Lite agents.
Runs N sequential passes, each building on the previous findings.
Zero LLM overhead for routing — uses agents/base_agent_lite.py directly.
"""

import sys, os, json, time
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#")[0].strip()
        if key:
            os.environ[key] = value

sys.path.insert(0, str(Path(__file__).parent))

from agents.base_agent_lite import BaseAgentLite
from core.contracts import ExecutionContract
from core.skill_registry import SkillRegistry
from core.hermes_client import HermesClient, Provider
from core.memory import AgentMemory


class CascadeResearch:
    """
    N-pass deep research. Each pass takes the previous output as context
    and drills deeper. No orchestrator overhead — agents run directly.
    """

    def __init__(self, topic: str, passes: int = 5, workspace: str = "workspace"):
        self.topic = topic
        self.passes = passes
        self.workspace = workspace
        Path(workspace).mkdir(exist_ok=True)

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.client = HermesClient(provider=Provider.DEEPSEEK, api_key=api_key)
        self.registry = SkillRegistry("skills")
        self.memory = AgentMemory(f"{workspace}/memory.db")

        self.agent = self._build_agent()
        self.findings: list[dict] = []
        self.total_tokens = 0

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"🔬 CASCADE RESEARCH: {self.topic}")
        print(f"   {self.passes} passes, building on each previous finding")
        print(f"{'='*60}\n")

        for i in range(1, self.passes + 1):
            query = self._build_query(i)
            contract = self._build_contract(i, query)

            print(f"\n{'─'*50}")
            print(f"📡 Pass {i}/{self.passes}")
            print(f"   Query: {query[:120]}...")
            print(f"{'─'*50}")

            t0 = time.time()
            result = self.agent.run(contract)
            elapsed = time.time() - t0

            self.total_tokens += result.tokens_used
            self.findings.append({
                "pass": i,
                "query": query,
                "success": result.success,
                "tokens": result.tokens_used,
                "elapsed_s": round(elapsed, 1),
                "output": result.output,
            })

            status = "✓" if result.success else "✗"
            summary = str(result.output.get("summary", result.output.get("content", "")))[:200]
            print(f"\n   [{status}] Pass {i} complete — {result.tokens_used} tok, {elapsed:.1f}s")
            print(f"   Key finding: {summary}")
            print(f"{'─'*50}\n")

        # Final synthesis
        synthesis = self._synthesize()
        self._save_report()

        print(f"\n{'='*60}")
        print(f"✅ RESEARCH COMPLETE")
        print(f"   Total tokens: {self.total_tokens}")
        print(f"   Total time: {sum(f['elapsed_s'] for f in self.findings):.1f}s")
        print(f"   Successful passes: {sum(1 for f in self.findings if f['success'])}/{self.passes}")
        print(f"{'='*60}\n")

        return {
            "topic": self.topic,
            "passes": self.findings,
            "synthesis": synthesis,
            "total_tokens": self.total_tokens,
            "total_elapsed_s": round(sum(f["elapsed_s"] for f in self.findings), 1),
            "passes_succeeded": sum(1 for f in self.findings if f["success"]),
        }

    def _build_query(self, pass_num: int) -> str:
        """Build a deep-dive query for each pass, layering on prior findings."""
        if pass_num == 1:
            return (
                f"Research the current state of {self.topic}. "
                f"Find the most important fundamental principles, recent breakthroughs (2024-2026), "
                f"and key open questions. Cite sources."
            )
        elif pass_num == 2:
            return (
                f"Building on the fundamentals of {self.topic}, research "
                f"the ROLE OF COMPUTATION — what computational methods (quantum computing, "
                f"classical simulation, tensor networks, machine learning) are driving "
                f"discoveries in this field. How does compute connect to the physics?"
            )
        elif pass_num == 3:
            return (
                f"Deep dive into COMPUTATIONAL METHODS + {self.topic}. "
                f"Which specific algorithms, hardware developments (quantum processors, GPUs, "
                f"neuromorphic), and software frameworks are advancing the field? "
                f"What are the computational bottlenecks? Cite recent papers."
            )
        elif pass_num == 4:
            return (
                f"Based on prior findings about {self.topic} and its computational methods, "
                f"research the PRACTICAL APPLICATIONS and THEORETICAL IMPLICATIONS. "
                f"How does the interplay of physics and computation change our understanding "
                f"of reality? What new phenomena or technologies emerge?"
            )
        else:  # pass 5
            return (
                f"Synthesize all prior research on {self.topic}. "
                f"Write a comprehensive summary covering: "
                f"(1) fundamental principles, (2) computational methods driving progress, "
                f"(3) recent breakthroughs, (4) practical applications and implications, "
                f"(5) a FORWARD-LOOKING VISION: how will computation and physics converge "
                f"in the next 5-10 years? What does this mean for humanity and technology?"
            )

    def _build_contract(self, pass_num: int, query: str) -> ExecutionContract:
        """Build a research contract with generous budget for deep search."""
        # Pass 5 (synthesis) gets read_file + write_file instead of web search
        if pass_num == self.passes:
            allowed = ["read_file", "write_file"]
        else:
            allowed = ["web_search", "http_get", "read_file"]

        return ExecutionContract(
            task_id=f"pass{pass_num:03d}",
            agent_id="agent_research",
            description=query,
            max_tokens=4000,
            max_tool_calls=4,
            allowed_tools=allowed,
            required_output_keys=["summary", "key_insights", "sources"],
            completion_conditions=["no_error_flag"],
            output_path=f"{self.workspace}/cascade/pass_{pass_num}.json",
        )

    def _build_agent(self) -> BaseAgentLite:
        class CascadeAgent(BaseAgentLite):
            agent_id = "agent_research"
            system_prompt = (
                "You are a deep research agent. You perform structured multi-pass "
                "research on complex scientific topics. Each pass builds on the previous one. "
                "Use web_search to find current information. Use http_get to fetch detailed content. "
                "Always cite your sources with URLs. "
                "Return JSON with 'summary' (2-3 paragraphs), 'key_insights' (3-5 bullet points), "
                "and 'sources' (list of URLs with descriptions)."
            )

        return CascadeAgent(
            client=self.client,
            registry=self.registry,
            memory=self.memory,
            workspace=self.workspace,
        )

    def _synthesize(self) -> str:
        """Build a text synthesis from all pass findings."""
        lines = [f"# Cascade Research: {self.topic}", ""]
        for f in self.findings:
            if f["success"]:
                output = f["output"]
                summary = output.get("summary", output.get("content", "No output"))
                insights = output.get("key_insights", output.get("key_points", []))
                lines.append(f"## Pass {f['pass']}: {f['query'][:100]}...\n")
                lines.append(summary + "\n")
                if insights:
                    lines.append("**Key insights:**")
                    for ins in insights:
                        lines.append(f"- {ins}")
                    lines.append("")
                lines.append(f"_(Tokens: {f['tokens']}, Time: {f['elapsed_s']}s)_\n")
        return "\n".join(lines)

    def _save_report(self):
        out = Path(self.workspace) / "cascade" / "research_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self._synthesize())
        json_out = Path(self.workspace) / "cascade" / "research_report.json"
        json_out.write_text(json.dumps({
            "topic": self.topic,
            "passes": self.findings,
            "total_tokens": self.total_tokens,
        }, indent=2))
        print(f"📄 Report saved to {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-pass cascade research with Lite agents")
    parser.add_argument("topic", nargs="?", default="quantum physics and the role of computation",
                        help="Research topic")
    parser.add_argument("--passes", type=int, default=5, help="Number of research passes")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    args = parser.parse_args()

    research = CascadeResearch(topic=args.topic, passes=args.passes, workspace=args.workspace)
    result = research.run()

    print("\n📋 FINAL SUMMARY")
    print(json.dumps({
        "topic": result["topic"],
        "total_tokens": result["total_tokens"],
        "total_time_s": result["total_elapsed_s"],
        "passes_succeeded": f"{result['passes_succeeded']}/{args.passes}",
    }, indent=2))
