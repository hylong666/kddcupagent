from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import PLAN_SOLVE_SYSTEM_PROMPT, TASK_ANALYSIS_SYSTEM_PROMPT
from data_agent_baseline.benchmark.schema import PublicTask


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Plan response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Plan response must be a JSON object.")
    return payload


@dataclass(frozen=True, slots=True)
class PlanSolveResult:
    plan: list[str]
    focus_tools: list[str]
    risks: list[str]
    raw_response: str


@dataclass(frozen=True, slots=True)
class TaskAnalysisResult:
    intent: str
    subtasks: list[str]
    constraints: list[str]
    raw_response: str


def parse_plan_solve_response(raw_response: str) -> PlanSolveResult:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    raw_plan = payload.get("plan", [])
    raw_tools = payload.get("focus_tools", [])
    raw_risks = payload.get("risks", [])

    if not isinstance(raw_plan, list) or not all(isinstance(item, str) for item in raw_plan):
        raise ValueError("plan must be a list of strings.")
    if not isinstance(raw_tools, list) or not all(isinstance(item, str) for item in raw_tools):
        raw_tools = []
    if not isinstance(raw_risks, list) or not all(isinstance(item, str) for item in raw_risks):
        raw_risks = []

    plan = [step.strip() for step in raw_plan if step.strip()]
    if not plan:
        raise ValueError("plan must not be empty.")

    focus_tools = [name.strip() for name in raw_tools if name.strip()]
    risks = [risk.strip() for risk in raw_risks if risk.strip()]
    return PlanSolveResult(plan=plan, focus_tools=focus_tools, risks=risks, raw_response=raw_response)


def parse_task_analysis_response(raw_response: str) -> TaskAnalysisResult:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    raw_intent = payload.get("intent", "")
    raw_subtasks = payload.get("subtasks", [])
    raw_constraints = payload.get("constraints", [])

    if not isinstance(raw_intent, str) or not raw_intent.strip():
        raise ValueError("intent must be a non-empty string.")
    if not isinstance(raw_subtasks, list) or not all(isinstance(item, str) for item in raw_subtasks):
        raise ValueError("subtasks must be a list of strings.")
    if not isinstance(raw_constraints, list) or not all(isinstance(item, str) for item in raw_constraints):
        raw_constraints = []

    subtasks = [item.strip() for item in raw_subtasks if item.strip()]
    if not subtasks:
        raise ValueError("subtasks must not be empty.")

    constraints = [item.strip() for item in raw_constraints if item.strip()]
    return TaskAnalysisResult(
        intent=raw_intent.strip(),
        subtasks=subtasks,
        constraints=constraints,
        raw_response=raw_response,
    )


def render_task_analysis(task_analysis: TaskAnalysisResult | None) -> str:
    if task_analysis is None:
        return ""

    lines = [
        "Task analysis:",
        f"- Inferred intent: {task_analysis.intent}",
        "- Decomposed subtasks:",
    ]
    lines.extend(f"  - {item}" for item in task_analysis.subtasks)
    if task_analysis.constraints:
        lines.append("- Constraints:")
        lines.extend(f"  - {item}" for item in task_analysis.constraints)
    return "\n".join(lines)


def generate_task_analysis(
    *,
    model: ModelAdapter,
    task: PublicTask,
    tool_descriptions: str,
    memory_snippets: str | None = None,
    system_prompt: str | None = None,
) -> TaskAnalysisResult:
    base_prompt = (system_prompt or TASK_ANALYSIS_SYSTEM_PROMPT).strip()

    memory_block = ""
    if memory_snippets:
        memory_block = f"\n\nLong-term memory (may be helpful, may be irrelevant):\n{memory_snippets.strip()}"

    system_content = (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        "Return exactly one ```json fenced block with one JSON object."
        f"{memory_block}"
    )
    user_content = (
        f"Question: {task.question}\n"
        "Infer the task intent, decompose the task, and capture any output constraints that should shape planning."
    )

    messages = [
        ModelMessage(role="system", content=system_content),
        ModelMessage(role="user", content=user_content),
    ]

    raw_response = model.complete(messages)
    return parse_task_analysis_response(raw_response)


def generate_plan_solve(
    *,
    model: ModelAdapter,
    task: PublicTask,
    tool_descriptions: str,
    task_analysis: TaskAnalysisResult | None = None,
    improvement_hints: list[str] | None = None,
    system_prompt: str | None = None,
) -> PlanSolveResult:
    base_prompt = (system_prompt or PLAN_SOLVE_SYSTEM_PROMPT).strip()

    improvement_block = ""
    if improvement_hints:
        rendered_improvements = "\n".join(f"- {item}" for item in improvement_hints if item.strip())
        if rendered_improvements:
            improvement_block = (
                "\n\nReusable checks for this task:\n"
                f"{rendered_improvements}"
            )

    analysis_block = ""
    rendered_task_analysis = render_task_analysis(task_analysis)
    if rendered_task_analysis:
        analysis_block = f"\n\nUpstream task analysis:\n{rendered_task_analysis}"

    system_content = (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        "Return exactly one ```json fenced block with one JSON object."
        f"{improvement_block}"
        f"{analysis_block}"
    )

    user_content = (
        f"Question: {task.question}\n"
        "Produce a concise Plan-Solve plan that uses the tools effectively."
    )

    messages = [
        ModelMessage(role="system", content=system_content),
        ModelMessage(role="user", content=user_content),
    ]

    raw_response = model.complete(messages)
    return parse_plan_solve_response(raw_response)


def render_memory_snippets(items: list[dict[str, Any]] | list[object]) -> str:
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            tags = ", ".join(str(tag) for tag in item.get("tags", []) if tag)
            lesson = str(item.get("lesson", "")).strip()
        else:
            tags = ""
            lesson = str(getattr(item, "lesson", "")).strip()
            raw_tags = getattr(item, "tags", [])
            if isinstance(raw_tags, list):
                tags = ", ".join(str(tag) for tag in raw_tags if tag)
        if not lesson:
            continue
        if tags:
            lines.append(f"- [{tags}] {lesson}")
        else:
            lines.append(f"- {lesson}")
    return "\n".join(lines)
