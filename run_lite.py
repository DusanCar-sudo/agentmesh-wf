#!/usr/bin/env python3
"""
AgentMesh · run_lite.py
Lean CLI entry point. Uses OrchestratorLite for token-efficient execution.

Usage:
  python run_lite.py "Research X and write a summary"
  python run_lite.py --status
  python run_lite.py --test
  python run_lite.py --learning --batch-size 5 "Research X"
"""

import sys
import os
import json
from pathlib import Path

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
            value = value.split("#")[0].strip()  # strip inline comments
            if key:
                os.environ[key] = value

_load_env()

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator_lite import OrchestratorLite
from core.hermes_client import Provider


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AgentMesh Lite — lean multi-agent orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_lite.py "Research AgentMesh and write a summary"
  python run_lite.py --status
  python run_lite.py --test
  python run_lite.py --learning "Research X and write report"
        """,
    )
    parser.add_argument("goal", nargs="?", help="Goal to execute")
    parser.add_argument("--status", action="store_true", help="Show memory stats")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--learning", action="store_true", help="Enable batched learning")
    parser.add_argument("--batch-size", type=int, default=5, help="Tasks between learning cycles")
    parser.add_argument("--provider", default="deepseek", help="LLM provider")
    parser.add_argument("--stream", action="store_true", help="Stream final responses")
    args = parser.parse_args()

    provider_map = {
        "deepseek": Provider.DEEPSEEK,
        "groq": Provider.GROQ,
        "openrouter": Provider.OPENROUTER,
        "anthropic": Provider.ANTHROPIC,
        "gemini": Provider.GEMINI,
        "together": Provider.TOGETHER,
    }
    provider = provider_map.get(args.provider.lower(), Provider.DEEPSEEK)

    orch = OrchestratorLite(
        provider=provider,
        stream=args.stream,
        enable_learning=args.learning,
        learning_batch_size=args.batch_size,
    )

    if args.status:
        stats = orch.memory.stats()
        print(json.dumps(stats, indent=2))
    elif args.test:
        result = orch.run("What is AgentMesh? Give a brief overview.")
        print("\n" + "=" * 60)
        print(json.dumps(result, indent=2))
    elif args.goal:
        result = orch.run(args.goal)
        print("\n" + "=" * 60)
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
