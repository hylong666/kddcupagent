from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.prompt import (
    PLAN_SOLVE_SYSTEM_PROMPT,
    PSRR_REACT_SYSTEM_PROMPT,
    RESPONSE_EXAMPLES,
    build_observation_prompt,
    build_task_prompt,
)
from data_agent_baseline.tools.registry import create_default_tool_registry


TOKEN_CHARS_PER_TOKEN = 4.0
EXTRA_TOKENS_PER_MODEL_CALL_LOW = 250
EXTRA_TOKENS_PER_MODEL_CALL_HIGH = 800
BRANCH_TARGET_TOOLS = {"read_csv", "read_json", "read_doc", "inspect_sqlite_schema"}

SC_POLICY_BLOCK = (
    "\n\nCritical-step interpretation policy:\n"
    "- For designated read or inspection tools, generate 2-3 distinct candidate actions.\n"
    "- Execute the candidates, compare observations, and keep the strongest branch.\n"
    "- Prefer branches that succeed and return non-empty evidence.\n"
)


@dataclass(slots=True)
class TaskBehavior:
    task_id: str
    succeeded: bool
    final_step_index: int
    step_count: int
    branch_generation_call_count: int
    elapsed_seconds: float | None
    total_tokens: int | None
    estimated_input_tokens_lower_bound: float
    estimated_output_tokens: float
    estimated_total_tokens_lower_bound: float
    estimated_total_tokens_low: float
    estimated_total_tokens_high: float
    data_formats: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TOOL_DESCRIPTIONS = create_default_tool_registry().describe_for_prompt()
SYSTEM_PROMPT_WITH_TOOLS = (
    f"{PSRR_REACT_SYSTEM_PROMPT.strip()}{SC_POLICY_BLOCK}\n\n"
    "Available tools:\n"
    f"{TOOL_DESCRIPTIONS}\n\n"
    f"{RESPONSE_EXAMPLES}\n\n"
    "You must always return a single ```json fenced block containing one JSON object "
    "with keys `thought`, `action`, and `action_input`, and no extra text."
)


def estimate_tokens(text: str) -> float:
    return len(text) / TOKEN_CHARS_PER_TOKEN


def load_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_suffix(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "no_extension"


def collect_context_formats(context_dir: Path) -> list[str]:
    if not context_dir.exists():
        return []
    formats: set[str] = set()
    for path in context_dir.rglob("*"):
        if path.is_file():
            formats.add(normalize_suffix(path))
    return sorted(formats)


def extract_total_tokens(trace_payload: dict[str, Any]) -> int | None:
    token_keys = ("total_tokens", "prompt_tokens", "completion_tokens")

    def visit(node: Any) -> int | None:
        if isinstance(node, dict):
            total_tokens = node.get("total_tokens")
            if isinstance(total_tokens, int):
                return total_tokens

            prompt_tokens = node.get("prompt_tokens")
            completion_tokens = node.get("completion_tokens")
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                return prompt_tokens + completion_tokens

            for key in token_keys:
                value = node.get(key)
                if isinstance(value, dict):
                    nested = visit(value)
                    if nested is not None:
                        return nested

            for value in node.values():
                nested = visit(value)
                if nested is not None:
                    return nested

        if isinstance(node, list):
            for item in node:
                nested = visit(item)
                if nested is not None:
                    return nested

        return None

    return visit(trace_payload)


def estimate_task_tokens(task_id: str, question: str, trace_payload: dict[str, Any]) -> dict[str, float | int]:
    steps = trace_payload.get("steps", [])
    history_messages: list[str] = []
    input_tokens = 0.0
    output_tokens = 0.0
    branch_generation_call_count = 0

    task_prompt = build_task_prompt(type("TaskPromptProxy", (), {"question": question})())
    for step in steps:
        prompt_text = SYSTEM_PROMPT_WITH_TOOLS + task_prompt + "".join(history_messages)
        input_tokens += estimate_tokens(prompt_text)

        raw_response = str(step.get("raw_response", "") or "")
        output_tokens += estimate_tokens(raw_response)
        history_messages.append(raw_response)
        history_messages.append(build_observation_prompt(step.get("observation", {})))
        if step.get("action") in BRANCH_TARGET_TOOLS:
            branch_generation_call_count += 1

    plan_user_prompt = f"Task: {task_id}\nQuestion: {question}\n\nAvailable tools:\n{TOOL_DESCRIPTIONS}"
    plan_input_tokens = estimate_tokens(PLAN_SOLVE_SYSTEM_PROMPT) + estimate_tokens(plan_user_prompt)

    model_call_count_lower_bound = len(steps) + 1
    model_call_count_estimated = model_call_count_lower_bound + branch_generation_call_count
    total_tokens_lower_bound = input_tokens + output_tokens + plan_input_tokens
    estimated_total_tokens_low = total_tokens_lower_bound + (
        model_call_count_estimated * EXTRA_TOKENS_PER_MODEL_CALL_LOW
    )
    estimated_total_tokens_high = total_tokens_lower_bound + (
        model_call_count_estimated * EXTRA_TOKENS_PER_MODEL_CALL_HIGH
    )

    return {
        "branch_generation_call_count": branch_generation_call_count,
        "estimated_input_tokens_lower_bound": input_tokens + plan_input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens_lower_bound": total_tokens_lower_bound,
        "estimated_total_tokens_low": estimated_total_tokens_low,
        "estimated_total_tokens_high": estimated_total_tokens_high,
    }


def analyze_task(run_dir: Path, input_root: Path, task_id: str) -> TaskBehavior:
    trace_path = run_dir / task_id / "trace.json"
    trace_payload = load_json(trace_path) if trace_path.exists() else {}
    task_payload = load_json(input_root / task_id / "task.json")
    steps = trace_payload.get("steps", [])
    final_step_index = max((int(step.get("step_index", 0)) for step in steps), default=0)
    context_dir = input_root / task_id / "context"
    token_estimates = estimate_task_tokens(task_id, str(task_payload.get("question", "")), trace_payload)
    return TaskBehavior(
        task_id=task_id,
        succeeded=bool(trace_payload.get("succeeded")),
        final_step_index=final_step_index,
        step_count=len(steps),
        branch_generation_call_count=int(token_estimates["branch_generation_call_count"]),
        elapsed_seconds=trace_payload.get("e2e_elapsed_seconds"),
        total_tokens=extract_total_tokens(trace_payload),
        estimated_input_tokens_lower_bound=float(token_estimates["estimated_input_tokens_lower_bound"]),
        estimated_output_tokens=float(token_estimates["estimated_output_tokens"]),
        estimated_total_tokens_lower_bound=float(token_estimates["estimated_total_tokens_lower_bound"]),
        estimated_total_tokens_low=float(token_estimates["estimated_total_tokens_low"]),
        estimated_total_tokens_high=float(token_estimates["estimated_total_tokens_high"]),
        data_formats=collect_context_formats(context_dir),
    )


def summarize_tasks(tasks: list[TaskBehavior]) -> dict[str, Any]:
    task_count = len(tasks)
    avg_final_step_index = sum(task.final_step_index for task in tasks) / task_count if task_count else 0.0
    avg_step_count = sum(task.step_count for task in tasks) / task_count if task_count else 0.0
    avg_branch_generation_calls = (
        sum(task.branch_generation_call_count for task in tasks) / task_count if task_count else 0.0
    )

    elapsed_values = [task.elapsed_seconds for task in tasks if isinstance(task.elapsed_seconds, (int, float))]
    avg_elapsed_seconds = sum(elapsed_values) / len(elapsed_values) if elapsed_values else None

    token_values = [task.total_tokens for task in tasks if isinstance(task.total_tokens, int)]
    avg_total_tokens = sum(token_values) / len(token_values) if token_values else None
    avg_estimated_input_tokens_lower_bound = (
        sum(task.estimated_input_tokens_lower_bound for task in tasks) / task_count if task_count else 0.0
    )
    avg_estimated_output_tokens = (
        sum(task.estimated_output_tokens for task in tasks) / task_count if task_count else 0.0
    )
    avg_estimated_total_tokens_lower_bound = (
        sum(task.estimated_total_tokens_lower_bound for task in tasks) / task_count if task_count else 0.0
    )
    avg_estimated_total_tokens_low = (
        sum(task.estimated_total_tokens_low for task in tasks) / task_count if task_count else 0.0
    )
    avg_estimated_total_tokens_high = (
        sum(task.estimated_total_tokens_high for task in tasks) / task_count if task_count else 0.0
    )
    total_estimated_input_tokens_lower_bound = sum(task.estimated_input_tokens_lower_bound for task in tasks)
    total_estimated_output_tokens = sum(task.estimated_output_tokens for task in tasks)
    total_estimated_total_tokens_lower_bound = sum(task.estimated_total_tokens_lower_bound for task in tasks)
    estimated_total_tokens_low = sum(task.estimated_total_tokens_low for task in tasks)
    estimated_total_tokens_high = sum(task.estimated_total_tokens_high for task in tasks)

    by_format: dict[str, list[TaskBehavior]] = defaultdict(list)
    by_format_combination: dict[str, list[TaskBehavior]] = defaultdict(list)
    format_combinations = Counter()
    for task in tasks:
        combination_key = "+".join(task.data_formats) if task.data_formats else "unknown"
        format_combinations[combination_key] += 1
        by_format_combination[combination_key].append(task)
        for data_format in task.data_formats:
            by_format[data_format].append(task)
        if not task.data_formats:
            by_format["unknown"].append(task)

    format_success = {}
    for data_format, group in sorted(by_format.items()):
        success_count = sum(1 for task in group if task.succeeded)
        format_success[data_format] = {
            "task_count": len(group),
            "success_count": success_count,
            "success_rate": success_count / len(group) if group else 0.0,
            "avg_final_step_index": sum(task.final_step_index for task in group) / len(group) if group else 0.0,
            "avg_elapsed_seconds": (
                sum(task.elapsed_seconds for task in group if isinstance(task.elapsed_seconds, (int, float)))
                / len([task for task in group if isinstance(task.elapsed_seconds, (int, float))])
                if any(isinstance(task.elapsed_seconds, (int, float)) for task in group)
                else None
            ),
        }

    format_combination_success = {}
    for combination_key, group in sorted(by_format_combination.items()):
        success_count = sum(1 for task in group if task.succeeded)
        format_combination_success[combination_key] = {
            "task_count": len(group),
            "success_count": success_count,
            "success_rate": success_count / len(group) if group else 0.0,
            "avg_final_step_index": sum(task.final_step_index for task in group) / len(group) if group else 0.0,
            "avg_elapsed_seconds": (
                sum(task.elapsed_seconds for task in group if isinstance(task.elapsed_seconds, (int, float)))
                / len([task for task in group if isinstance(task.elapsed_seconds, (int, float))])
                if any(isinstance(task.elapsed_seconds, (int, float)) for task in group)
                else None
            ),
        }

    return {
        "task_count": task_count,
        "avg_steps_per_task": avg_final_step_index,
        "avg_recorded_steps_per_task": avg_step_count,
        "avg_branch_generation_calls_per_task": avg_branch_generation_calls,
        "avg_elapsed_seconds": avg_elapsed_seconds,
        "avg_total_tokens": avg_total_tokens,
        "token_metric_available": avg_total_tokens is not None,
        "avg_estimated_input_tokens_lower_bound": avg_estimated_input_tokens_lower_bound,
        "avg_estimated_output_tokens": avg_estimated_output_tokens,
        "avg_estimated_total_tokens_lower_bound": avg_estimated_total_tokens_lower_bound,
        "total_estimated_input_tokens_lower_bound": total_estimated_input_tokens_lower_bound,
        "total_estimated_output_tokens": total_estimated_output_tokens,
        "total_estimated_total_tokens_lower_bound": total_estimated_total_tokens_lower_bound,
        "estimated_avg_total_tokens_range": [
            avg_estimated_total_tokens_low,
            avg_estimated_total_tokens_high,
        ],
        "estimated_total_tokens_range_for_run": [
            estimated_total_tokens_low,
            estimated_total_tokens_high,
        ],
        "format_combination_distribution": dict(format_combinations),
        "success_by_data_format": format_success,
        "success_by_format_combination": format_combination_success,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze steps, time, and format-level adaptability for a run.")
    parser.add_argument("run_dir", type=Path, help="Run directory under artifacts/runs")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/public/input"),
        help="Task input root containing task_x/context directories",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for the full analysis JSON",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_ids = sorted(path.name for path in args.run_dir.iterdir() if path.is_dir() and path.name.startswith("task_"))
    tasks = [analyze_task(args.run_dir, args.input_root, task_id) for task_id in task_ids]
    payload = {
        "run_dir": str(args.run_dir),
        "input_root": str(args.input_root),
        "summary": summarize_tasks(tasks),
        "tasks": [task.to_dict() for task in tasks],
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()