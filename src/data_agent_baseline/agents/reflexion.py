from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.memory import MemoryItem
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import REFLEXION_SYSTEM_PROMPT
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import PublicTask


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
            raise ValueError("Reflexion response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Reflexion response must be a JSON object.")
    return payload


@dataclass(frozen=True, slots=True)
class ReflexionResult:
    reflection: str
    improvements: list[str]
    memory_items: list[MemoryItem]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflection": self.reflection,
            "improvements": list(self.improvements),
            "memory_items": [item.to_dict() for item in self.memory_items],
            "raw_response": self.raw_response,
        }


def parse_reflexion_response(
    *,
    raw_response: str,
    task_id: str,
    succeeded: bool,
) -> ReflexionResult:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    reflection = payload.get("reflection", "")
    improvements = payload.get("improvements", [])
    raw_memory = payload.get("memory_items", [])

    if not isinstance(reflection, str):
        reflection = ""
    if not isinstance(improvements, list) or not all(isinstance(item, str) for item in improvements):
        improvements = []
    if not isinstance(raw_memory, list):
        raw_memory = []

    memory_items: list[MemoryItem] = []
    for candidate in raw_memory[:5]:
        if not isinstance(candidate, dict):
            continue
        tags = candidate.get("tags", [])
        lesson = candidate.get("lesson", "")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            tags = []
        if not isinstance(lesson, str):
            continue
        cleaned_lesson = lesson.strip()
        if not cleaned_lesson:
            continue
        created_at = candidate.get("created_at")
        created_at_value = str(created_at).strip() if isinstance(created_at, str) else ""
        if not created_at_value:
            created_at_value = _now_iso()
        memory_items.append(
            MemoryItem(
                created_at=created_at_value,
                tags=[tag.strip() for tag in tags if tag.strip()][:8],
                lesson=cleaned_lesson[:800],
                source_task_id=task_id,
                succeeded=succeeded,
            )
        )

    return ReflexionResult(
        reflection=reflection.strip()[:4000],
        improvements=[item.strip() for item in improvements if item.strip()][:12],
        memory_items=memory_items,
        raw_response=raw_response,
    )


def generate_reflexion(
    *,
    model: ModelAdapter,
    task: PublicTask,
    run_result: AgentRunResult,
    plan: list[str] | None = None,
    system_prompt: str | None = None,
) -> ReflexionResult:
    base_prompt = (system_prompt or REFLEXION_SYSTEM_PROMPT).strip()

    plan_block = ""
    if plan:
        rendered = "\n".join(f"- {step}" for step in plan)
        plan_block = f"\n\nPlan used:\n{rendered}\n"

    steps_summary = []
    for step in run_result.steps:
        observation = step.observation or {}
        if "tool" in observation:
            summary = f"{step.step_index}. {step.action} -> ok={step.ok}"
        elif step.action == "__error__":
            summary = f"{step.step_index}. error -> {observation.get('error', '')}"
        else:
            summary = f"{step.step_index}. {step.action} -> ok={step.ok}"
        steps_summary.append(summary)

    user_content = (
        f"Task: {task.task_id}\n"
        f"Question: {task.question}\n"
        f"Succeeded: {run_result.succeeded}\n"
        f"Failure reason: {run_result.failure_reason or ''}\n"
        f"Step summary:\n" + "\n".join(steps_summary[:80]) + plan_block
    )

    messages = [
        ModelMessage(role="system", content=base_prompt),
        ModelMessage(role="user", content=user_content),
    ]
    raw_response = model.complete(messages)
    return parse_reflexion_response(
        raw_response=raw_response,
        task_id=task.task_id,
        succeeded=run_result.succeeded,
    )


def store_reflexion(task_output_dir: Path, reflexion: ReflexionResult) -> Path:
    task_output_dir.mkdir(parents=True, exist_ok=True)
    path = task_output_dir / "reflexion.json"
    path.write_text(json.dumps(reflexion.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
