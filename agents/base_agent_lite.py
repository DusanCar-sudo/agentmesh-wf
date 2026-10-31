"""
AgentMesh · agents/base_agent_lite.py
Lean agent loop — max 3 tool iterations, simplified system prompt, no Honcho.

Token savings:
  Original: up to 9 LLM calls with growing message history
  Lite:     max 3 LLM calls, compact messages
"""

from __future__ import annotations
import json, time, traceback
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.contracts import ExecutionContract, ContractResult, ConditionChecker
from core.state_manager import FileBackedState, AgentStatus
from core.skill_registry import SkillRegistry, SkillError
from core.hermes_client import HermesClient
from core.memory import AgentMemory, Episode, make_episode_id


class BaseAgentLite:
    """
    Lean agent loop. Override system_prompt in subclass.
    No Honcho bridge, no per-task learning, no complex state tracking.
    """

    system_prompt: str = "You are a helpful AI agent."
    agent_id: str = "base_agent"

    def __init__(
        self,
        client: HermesClient,
        registry: SkillRegistry,
        memory: AgentMemory,
        workspace: str = "workspace",
        stream: bool = False,
    ):
        self.client = client
        self.registry = registry
        self.memory = memory
        self.workspace = workspace
        self.stream = stream

    def run(self, contract: ExecutionContract) -> ContractResult:
        """
        Execute a task. Lean loop: max 3 tool iterations, no tracing overhead.
        """
        t0 = time.time()
        contract.save(self.workspace)

        prompt = self._build_system_prompt(contract)
        tools = self.registry.get_openai_tools(contract.allowed_tools)

        messages = [{"role": "user", "content": self._task_message(contract)}]
        output = {}
        error = None
        tool_calls_made = 0
        tokens_used = 0

        try:
            # Lean loop: max 3 iterations
            for iteration in range(contract.max_tool_calls + 1):
                resp = self.client.call_with_tools(
                    messages=messages,
                    tools=tools,
                    system=prompt,
                    max_tokens=max(100, min(contract.max_tokens - tokens_used, 1200)),
                )
                tokens_used += resp.total_tokens

                # No tool call — model is done
                if not resp.has_tool_call:
                    content = resp.content
                    if self.stream:
                        content = self._stream_final(messages, prompt)
                    messages.append({"role": "assistant", "content": content})
                    output = self._parse_output(content, contract)
                    break

                # Execute tool calls
                tc_list = resp.tool_calls
                tool_call_msgs = [
                    {"id": f"call_{iteration}_{i}", "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for i, tc in enumerate(tc_list)
                ]
                messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": tool_call_msgs,
                })

                for i, tc in enumerate(tc_list):
                    call_id = f"call_{iteration}_{i}"
                    name = tc["name"]
                    args = tc["arguments"]

                    if tool_calls_made >= contract.max_tool_calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps({"error": "tool_budget_exceeded"}),
                        })
                        continue

                    if name not in contract.allowed_tools:
                        result = {"error": f"Tool '{name}' not permitted"}
                    else:
                        try:
                            result = self.registry.execute(name, args)
                        except SkillError as e:
                            result = {"error": str(e)}

                    tool_calls_made += 1
                    self.memory.record_skill_call(name, "error" not in result, resp.tokens_out)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result),
                    })

                # Force final answer after tool budget
                if tool_calls_made >= contract.max_tool_calls:
                    messages.append({"role": "user",
                        "content": "Tool budget exhausted. Write your final JSON answer now."})
                    resp = self.client.chat(messages=messages, system=prompt, max_tokens=800)
                    tokens_used += resp.total_tokens
                    output = self._parse_output(resp.content, contract)
                    break

        except Exception as e:
            error = str(e)

        # Completion check
        checker = ConditionChecker()
        all_passed, failed_conditions = checker.check_all(contract, output)
        success = all_passed and error is None

        elapsed = time.time() - t0
        result = ContractResult(
            task_id=contract.task_id,
            agent_id=self.agent_id,
            success=success,
            output=output,
            tokens_used=tokens_used,
            tool_calls_made=tool_calls_made,
            elapsed_seconds=round(elapsed, 2),
            error=error,
            failed_conditions=failed_conditions,
        )
        result.save(self.workspace)

        # Store episode (for future batched learning)
        episode = Episode(
            episode_id=make_episode_id(self.agent_id, contract.task_id),
            agent_id=self.agent_id,
            task_id=contract.task_id,
            task_description=contract.description,
            success=success,
            tokens_used=tokens_used,
            tool_calls=tool_calls_made,
            elapsed_seconds=elapsed,
            output_summary=str(output)[:300],
            lessons=[],
            traces=[],
        )
        self.memory.store_episode(episode)

        status = "✓" if success else "✗"
        print(f"[{self.agent_id}] {status} {contract.task_id} "
              f"| {tokens_used}tok | {tool_calls_made} tools | {elapsed:.1f}s")

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(self, contract: ExecutionContract) -> str:
        parts = [self.system_prompt]

        # Inject lessons (lightweight, from memory only)
        lessons = self.memory.format_lessons_for_prompt(self.agent_id)
        if lessons:
            parts.append(lessons)

        # Contract block (budget, permissions, required output)
        parts.append(f"CONTRACT:\n{contract.to_prompt_block()}")

        return "\n\n".join(parts)

    def _task_message(self, contract: ExecutionContract) -> str:
        return (
            f"Complete this task: {contract.description}\n\n"
            f"Required output keys: {', '.join(contract.required_output_keys)}\n\n"
            f"Respond with a JSON object containing the required keys when done."
        )

    def _stream_final(self, messages: list[dict], system: str) -> str:
        print(f"\n[{self.agent_id}] Streaming response:\n")
        accumulated = []
        for token in self.client.stream(messages, system=system):
            print(token, end="", flush=True)
            accumulated.append(token)
        print()
        return "".join(accumulated)

    def _parse_output(self, content: str, contract: ExecutionContract) -> dict:
        if not content:
            return {}
        text = content.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if part.startswith("json"):
                    text = part[4:].strip()
                    break
                elif "{" in part:
                    text = part.strip()
                    break
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                out = contract.output_file()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(data, indent=2))
                return data
        except Exception:
            pass
        return {"content": content, "raw": True}
