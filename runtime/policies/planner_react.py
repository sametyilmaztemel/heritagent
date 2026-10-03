"""ReAct-style planner policy (G0 seed; issue #7 criterion 6; tool catalog per critic review).

The planner owns the reasoning format: it builds the model messages
(including only the currently expressed skills AND the tool catalog from
`env.list_tools()`, rendered deterministically) and parses ReAct-formatted
replies into typed step decisions. Parsing fails closed — the execution
loop's retry policy decides whether to reprompt.
"""

from __future__ import annotations

import json
import re

from runtime.model_adapters import ModelMessage, ToolSpec
from runtime.policies.decisions import StepDecision, ToolAction
from runtime.policies.errors import PlanParseError
from runtime.skills import skill_block

FORMAT_INSTRUCTIONS = """Respond with exactly ONE step in one of these formats:

Thought: <one sentence of reasoning>
Action: <tool name>
Action Input: <a single JSON object with the tool's arguments>

or, when the task is complete:

Thought: <one sentence of reasoning>
Final Answer: <the final answer>"""

_FINAL_RE = re.compile(r"Final Answer:\s*(.+)\s*$", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+)")
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_\-]+)\s*$", re.MULTILINE)
_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})\s*$", re.DOTALL)


def tool_catalog_block(tools: list[ToolSpec]) -> str:
    """Deterministic, evaluation-safe rendering of the offered tool catalog
    (names, descriptions, argument JSON Schemas). Only `env.list_tools()`
    content enters here — no skill applicability or other evaluation-only
    metadata can appear."""
    if not tools:
        return "Available tools: (none)"
    lines = ["Available tools:"]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  Action Input JSON Schema: "
                     f"{json.dumps(tool.parameters_schema, sort_keys=True)}")
    return "\n".join(lines)


class ReActPlanner:
    def __init__(self, config: dict):
        if config.get("style") != "react":
            raise ValueError("ReActPlanner requires a 'react' style policy config")
        self.max_plan_steps = int(config["max_plan_steps"])

    def build_messages(self, *, task: str, observations: list[str], skills, tools: list[ToolSpec],
                       step: int, max_steps: int) -> list[ModelMessage]:
        system = (
            "You are a ReAct agent. Solve the task step by step using the available tools.\n"
            f"{FORMAT_INSTRUCTIONS}\n\n{tool_catalog_block(tools)}\n\n"
            f"Expressed skills (use when applicable):\n{skill_block(skills)}"
        )
        history = "\n".join(f"[{i}] {content}" for i, content in enumerate(observations))
        user = (f"Task: {task}\n\nObservations so far:\n{history}\n\n"
                f"Step {step} of at most {max_steps}. Provide the next single step.")
        return [ModelMessage(role="system", content=system), ModelMessage(role="user", content=user)]

    @staticmethod
    def parse(text: str) -> StepDecision:
        if "Final Answer:" in text:
            match = _FINAL_RE.search(text)
            if not match:
                raise PlanParseError("Final Answer present but empty")
            thought = _THOUGHT_RE.search(text)
            return StepDecision(thought=thought.group(1).strip() if thought else "",
                                action=None, final_answer=match.group(1).strip())
        thought_match = _THOUGHT_RE.search(text)
        action_match = _ACTION_RE.search(text)
        input_match = _INPUT_RE.search(text)
        if not (thought_match and action_match and input_match):
            raise PlanParseError("reply does not match the ReAct format (Thought/Action/Action Input or Final Answer)")
        try:
            arguments = json.loads(input_match.group(1))
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Action Input is not valid JSON: {exc}") from None
        if not isinstance(arguments, dict):
            raise PlanParseError("Action Input must be a JSON object")
        return StepDecision(thought=thought_match.group(1).strip(),
                            action=ToolAction(name=action_match.group(1), arguments=arguments),
                            final_answer=None)
