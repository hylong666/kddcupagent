from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry


BRANCH_GENERATION_PROMPT = """
You are generating multiple candidate next actions for a ReAct-style data agent.

Goal:
- Propose a small set of distinct tool-call candidates for the same next step.
- Keep them grounded in the task and the available tools.
- Focus on alternative read, inspection, or interpretation strategies before heavy execution.
- For file-reading work, vary the source choice, preview scope, or inspection strategy meaningfully.

Output format:
- Return exactly one JSON object in a single ```json fenced block.
- The JSON object must have a single key `candidates`.
- `candidates` must be a list of 2-3 objects.
- Each candidate object must contain keys `thought`, `action`, and `action_input`.

Do not include any extra text outside the fenced JSON.
""".strip()


BRANCH_SELECTION_PROMPT = """
You are evaluating multiple executed branch candidates for a single ReAct step.

Goal:
- Compare the observed tool results for the same step goal.
- Select the observation that best satisfies the step objective.
- Prefer observations that are most directly useful for the current step over generic or noisy outputs.
- If every observation is poor, still choose the least-bad option.

Output format:
- Return exactly one JSON object in a single ```json fenced block.
- The JSON object must contain exactly two keys: `selected_index` and `reason`.
- `selected_index` must be a 1-based integer referring to one of the provided candidates.
- `reason` must be a short sentence.

Do not include any extra text outside the fenced JSON.
""".strip()


@dataclass(frozen=True, slots=True)
class ReActSelfConsistencyConfig:
    max_steps: int = 16
    branch_candidates: int = 3
    branch_target_tools: tuple[str, ...] = field(
        default_factory=lambda: (
            "execute_python",
            "read_csv",
            "read_json",
            "read_doc",
            "inspect_sqlite_schema",
            "execute_context_sql",
        )
    )


@dataclass(frozen=True, slots=True)
class BranchExecutionResult:
    model_step: ModelStep
    observation: dict[str, Any]
    tool_result: ToolExecutionResult


def _is_branch_candidate_allowed(candidate_action: str, allowed_actions: tuple[str, ...]) -> bool:
    return candidate_action in set(allowed_actions)


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
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _parse_candidate_payload(raw_response: str) -> list[ModelStep]:
    payload = _load_single_json_object(_strip_json_fence(raw_response))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list.")

    parsed: list[ModelStep] = []
    for candidate in candidates[:3]:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be a JSON object.")
        parsed.append(
            parse_model_step(
                "```json\n"
                + json.dumps(candidate, ensure_ascii=False)
                + "\n```"
            )
        )
    return parsed


def _parse_branch_selection(raw_response: str, candidate_count: int) -> tuple[int, str]:
    payload = _load_single_json_object(_strip_json_fence(raw_response))
    selected_index = payload.get("selected_index")
    reason = payload.get("reason")
    if not isinstance(selected_index, int):
        raise ValueError("selected_index must be an integer.")
    if selected_index < 1 or selected_index > candidate_count:
        raise ValueError("selected_index is out of range.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    return selected_index, reason.strip()


class ReActSelfConsistencyAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActSelfConsistencyConfig | None = None,
        system_prompt: str | None = None,
        task_prompt_builder: Callable[[PublicTask, AgentRuntimeState], str] | None = None,
        step_callback=None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActSelfConsistencyConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.task_prompt_builder = task_prompt_builder
        self.step_callback = step_callback

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        task_prompt = build_task_prompt(task)
        if self.task_prompt_builder is not None:
            task_prompt = self.task_prompt_builder(task, state)
        messages.append(ModelMessage(role="user", content=task_prompt))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def _execute_model_step(
        self,
        task: PublicTask,
        *,
        model_step: ModelStep,
    ) -> BranchExecutionResult:
        tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
        observation = {
            "ok": tool_result.ok,
            "tool": model_step.action,
            "content": tool_result.content,
        }
        return BranchExecutionResult(
            model_step=model_step,
            observation=observation,
            tool_result=tool_result,
        )

    def _should_branch(self, model_step: ModelStep) -> bool:
        return model_step.action in set(self.config.branch_target_tools)

    def _generate_branch_candidates(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        proposed_step: ModelStep,
        branch_candidates: int,
    ) -> list[ModelStep]:
        messages = self._build_messages(task, state)
        prompt = (
            f"Original proposed next action:\n{proposed_step.raw_response}\n\n"
            f"Generate exactly {branch_candidates} candidate next actions for this same step. "
            "Keep one candidate close to the original proposal and make the others meaningfully different. "
            "Prefer alternative read, schema-inspection, or evidence-gathering strategies over heavy execution."
        )
        branch_messages = [
            ModelMessage(
                role="system",
                content=build_system_prompt(
                    self.tools.describe_for_prompt(),
                    system_prompt=BRANCH_GENERATION_PROMPT,
                ),
            ),
            *messages[1:],
            ModelMessage(role="user", content=prompt),
        ]
        raw_response = self.model.complete(branch_messages)
        candidates = _parse_candidate_payload(raw_response)

        deduped: list[ModelStep] = [proposed_step]
        seen = {
            json.dumps(
                {
                    "action": proposed_step.action,
                    "action_input": proposed_step.action_input,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        }
        for candidate in candidates:
            if not _is_branch_candidate_allowed(candidate.action, self.config.branch_target_tools):
                continue
            key = json.dumps(
                {
                    "action": candidate.action,
                    "action_input": candidate.action_input,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
            if len(deduped) >= branch_candidates:
                break
        return deduped

    def _build_branch_selection_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        proposed_step: ModelStep,
        branch_results: list[BranchExecutionResult],
    ) -> list[ModelMessage]:
        messages = self._build_messages(task, state)
        selection_payload = {
            "step_goal": {
                "thought": proposed_step.thought,
                "action": proposed_step.action,
                "action_input": proposed_step.action_input,
            },
            "candidates": [
                {
                    "index": index,
                    "thought": branch_result.model_step.thought,
                    "action": branch_result.model_step.action,
                    "action_input": branch_result.model_step.action_input,
                    "observation": branch_result.observation,
                    "is_terminal": branch_result.tool_result.is_terminal,
                }
                for index, branch_result in enumerate(branch_results, start=1)
            ],
        }
        prompt = (
            "Evaluate the executed branch candidates for the current step and choose the best observation.\n\n"
            f"Original proposed step:\n{proposed_step.raw_response}\n\n"
            "Executed candidates and observations:\n"
            f"```json\n{json.dumps(selection_payload, ensure_ascii=False, indent=2)}\n```"
        )
        return [
            ModelMessage(
                role="system",
                content=build_system_prompt(
                    self.tools.describe_for_prompt(),
                    system_prompt=BRANCH_SELECTION_PROMPT,
                ),
            ),
            *messages[1:],
            ModelMessage(role="user", content=prompt),
        ]

    def _fallback_branch_selection(self, branch_results: list[BranchExecutionResult]) -> tuple[int, str]:
        if not branch_results:
            return 1, "Selected the first branch because no branch results were available."

        for index, branch_result in enumerate(branch_results, start=1):
            if branch_result.tool_result.ok:
                return index, "Fallback selected the first successful branch after branch evaluation failed."

        return 1, "Fallback selected the first branch because branch evaluation failed and no branch succeeded."

    def _select_best_branch(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        proposed_step: ModelStep,
        branch_results: list[BranchExecutionResult],
    ) -> tuple[int, str]:
        if not branch_results:
            return 1, "Selected the first branch because no branch results were available."

        selection_messages = self._build_branch_selection_messages(
            task,
            state,
            proposed_step=proposed_step,
            branch_results=branch_results,
        )
        try:
            raw_response = self.model.complete(selection_messages)
            return _parse_branch_selection(raw_response, len(branch_results))
        except Exception:
            return self._fallback_branch_selection(branch_results)

    def _execute_step(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        step_index: int,
        model_step: ModelStep,
        raw_response: str,
    ) -> tuple[StepRecord, ToolExecutionResult]:
        if self._should_branch(model_step):
            return self.execute_step_with_branches(
                task,
                state,
                step_index=step_index,
                proposed_step=model_step,
            )

        execution_result = self._execute_model_step(task, model_step=model_step)
        return (
            StepRecord(
                step_index=step_index,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=raw_response,
                observation=execution_result.observation,
                ok=execution_result.tool_result.ok,
            ),
            execution_result.tool_result,
        )

    def execute_step_with_branches(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        step_index: int,
        proposed_step: ModelStep,
    ) -> tuple[StepRecord, ToolExecutionResult]:
        candidate_count = min(max(int(self.config.branch_candidates), 2), 3)
        try:
            branch_candidates = self._generate_branch_candidates(
                task,
                state,
                proposed_step=proposed_step,
                branch_candidates=candidate_count,
            )
        except Exception:
            branch_candidates = [proposed_step]

        branch_results: list[BranchExecutionResult] = []
        for branch_candidate in branch_candidates:
            try:
                branch_results.append(
                    self._execute_model_step(
                        task,
                        model_step=branch_candidate,
                    )
                )
            except Exception as exc:
                branch_results.append(
                    BranchExecutionResult(
                        model_step=branch_candidate,
                        observation={
                            "ok": False,
                            "tool": branch_candidate.action,
                            "error": str(exc),
                        },
                        tool_result=ToolExecutionResult(ok=False, content={"error": str(exc)}),
                    )
                )

        selected_index, selection_reason = self._select_best_branch(
            task,
            state,
            proposed_step=proposed_step,
            branch_results=branch_results,
        )
        selected_result = branch_results[selected_index - 1]
        observation = dict(selected_result.observation)
        observation["branching"] = {
            "candidate_count": len(branch_results),
            "selected_index": selected_index,
            "selection_reason": selection_reason,
            "candidates": [
                {
                    "index": index,
                    "action": branch_result.model_step.action,
                    "action_input": branch_result.model_step.action_input,
                    "observation": branch_result.observation,
                    "is_terminal": branch_result.tool_result.is_terminal,
                }
                for index, branch_result in enumerate(branch_results, start=1)
            ],
        }
        return (
            StepRecord(
                step_index=step_index,
                thought=selected_result.model_step.thought,
                action=selected_result.model_step.action,
                action_input=selected_result.model_step.action_input,
                raw_response=selected_result.model_step.raw_response,
                observation=observation,
                ok=selected_result.tool_result.ok,
            ),
            selected_result.tool_result,
        )

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState(step_callback=self.step_callback)
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
                step_record, tool_result = self._execute_step(
                    task,
                    state,
                    step_index=step_index,
                    model_step=model_step,
                    raw_response=raw_response,
                )
                state.append_step(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.append_step(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
