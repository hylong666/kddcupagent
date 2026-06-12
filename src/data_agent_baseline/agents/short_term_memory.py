from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.runtime import AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask


def _truncate_text(value: str, *, limit: int = 180) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _summarize_content(content: Any) -> str:
    if not isinstance(content, dict):
        return _truncate_text(str(content), limit=120)

    summary_parts: list[str] = []

    columns = content.get("columns")
    if isinstance(columns, list) and columns:
        rendered_columns = ", ".join(str(column) for column in columns[:4])
        summary_parts.append(f"columns={rendered_columns}")

    rows = content.get("rows")
    if isinstance(rows, list):
        summary_parts.append(f"rows={len(rows)}")

    row_count = content.get("row_count")
    if isinstance(row_count, int):
        summary_parts.append(f"row_count={row_count}")

    entries = content.get("entries")
    if isinstance(entries, list):
        summary_parts.append(f"entries={len(entries)}")

    tables = content.get("tables")
    if isinstance(tables, list):
        summary_parts.append(f"tables={len(tables)}")

    preview = content.get("preview")
    if isinstance(preview, str) and preview.strip():
        summary_parts.append(f"preview={_truncate_text(preview, limit=80)}")

    output = content.get("output")
    if isinstance(output, str) and output.strip():
        summary_parts.append(f"output={_truncate_text(output, limit=80)}")

    error = content.get("error")
    if isinstance(error, str) and error.strip():
        summary_parts.append(f"error={_truncate_text(error, limit=80)}")

    if summary_parts:
        return "; ".join(summary_parts)
    return _truncate_text(str(content), limit=120)


@dataclass(slots=True)
class TaskShortTermMemory:
    task_question: str
    max_recent_steps: int = 5
    plan_steps: list[str] = field(default_factory=list)
    total_steps: int = 0
    last_action: str | None = None
    recent_events: list[dict[str, str]] = field(default_factory=list, repr=False)

    @property
    def confirmed_evidence(self) -> list[str]:
        return [event["text"] for event in self.recent_events if event["kind"] == "evidence"]

    @property
    def recent_blockers(self) -> list[str]:
        return [event["text"] for event in self.recent_events if event["kind"] == "blocker"]

    @classmethod
    def from_task(
        cls,
        task: PublicTask,
        *,
        max_recent_steps: int = 5,
    ) -> "TaskShortTermMemory":
        return cls(task_question=task.question, max_recent_steps=max_recent_steps)

    @classmethod
    def replay(
        cls,
        *,
        task_question: str,
        steps: list[StepRecord | dict[str, Any]],
        plan_steps: list[str] | None = None,
        max_recent_steps: int = 5,
        succeeded: bool = False,
        failure_reason: str | None = None,
    ) -> "TaskShortTermMemory":
        memory = cls(
            task_question=task_question,
            max_recent_steps=max_recent_steps,
        )
        if plan_steps:
            memory.set_plan(plan_steps)

        state = AgentRuntimeState()
        for raw_step in steps:
            step = raw_step if isinstance(raw_step, StepRecord) else cls._coerce_step_record(raw_step)
            state.steps.append(step)
            memory.record_step(step, state)

        memory.finalize(succeeded=succeeded, failure_reason=failure_reason)
        return memory

    def set_plan(self, plan_steps: list[str]) -> None:
        self.plan_steps = [step.strip() for step in plan_steps if step.strip()]

    def finalize(self, *, succeeded: bool, failure_reason: str | None = None) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_question": self.task_question,
            "plan_steps": list(self.plan_steps),
            "confirmed_evidence": list(self.confirmed_evidence),
            "recent_blockers": list(self.recent_blockers),
        }

    def record_step(self, step: StepRecord, state: AgentRuntimeState) -> None:
        self.total_steps = len(state.steps)
        self.last_action = step.action

        if step.action == "answer":
            return

        if step.ok:
            event = {"kind": "evidence", "text": self._summarize_confirmed_evidence(step)}
        else:
            event = {"kind": "blocker", "text": self._summarize_blocker(step)}

        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent_steps:
            self.recent_events.pop(0)

    def build_execution_context(self) -> str:
        sections = [
            "Task-local execution summary for the current task only:",
            f"- Current task question: {self.task_question}",
        ]

        if self.plan_steps:
            sections.append("- Plan steps:")
            sections.extend(f"  - {step}" for step in self.plan_steps)
        if self.confirmed_evidence:
            sections.append("- Confirmed evidence from recent successful steps:")
            sections.extend(f"  - {item}" for item in self.confirmed_evidence)
        if self.recent_blockers:
            sections.append("- Recent blockers:")
            sections.extend(f"  - {item}" for item in self.recent_blockers)
        if not self.confirmed_evidence and not self.recent_blockers:
            sections.append("- Confirmed evidence from recent successful steps: none yet")
        return "\n".join(sections).strip()

    def _summarize_confirmed_evidence(self, step: StepRecord) -> str:
        observation = step.observation or {}
        outcome = _summarize_content(observation.get("content", observation))
        return f"{step.action} succeeded with {outcome}"

    def _summarize_blocker(self, step: StepRecord) -> str:
        observation = step.observation or {}
        error_text = _truncate_text(str(observation.get("error") or observation.get("content") or observation), limit=140)
        return f"{step.action} failed: {error_text}"

    @staticmethod
    def _coerce_step_record(payload: dict[str, Any]) -> StepRecord:
        return StepRecord(
            step_index=int(payload.get("step_index", 0)),
            thought=str(payload.get("thought", "")),
            action=str(payload.get("action", "")),
            action_input=dict(payload.get("action_input", {})) if isinstance(payload.get("action_input"), dict) else {},
            raw_response=str(payload.get("raw_response", "")),
            observation=dict(payload.get("observation", {})) if isinstance(payload.get("observation"), dict) else {},
            ok=bool(payload.get("ok", False)),
        )
