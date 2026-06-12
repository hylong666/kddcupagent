from __future__ import annotations

import csv
import json
import multiprocessing
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.memory import MemoryItem, LongTermMemory
from data_agent_baseline.agents.psrr_agent import PlanSolveReActReflexionAgent, PSRRAgentConfig
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.react_sc import ReActSelfConsistencyAgent, ReActSelfConsistencyConfig
from data_agent_baseline.agents.psrr_react_sc_agent import (
    PlanSolveReActSelfConsistencyAgent,
    PSRRSelfConsistencyAgentConfig,
)
from data_agent_baseline.agents.reflexion import ReflexionResult, generate_reflexion
from data_agent_baseline.agents.short_term_memory import TaskShortTermMemory
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import PROJECT_ROOT as REPO_ROOT
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STM_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "STM"
_MEMORY_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _build_partial_run_payload(
    *,
    task_id: str,
    steps: list[dict[str, Any]],
    answer: dict[str, Any] | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": answer,
        "steps": steps,
        "failure_reason": failure_reason,
        "succeeded": answer is not None and failure_reason is None,
    }


def _write_partial_trace(
    *,
    trace_path: Path,
    task_id: str,
    steps: list[dict[str, Any]],
    answer: dict[str, Any] | None = None,
    failure_reason: str | None = None,
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        trace_path,
        _build_partial_run_payload(
            task_id=task_id,
            steps=steps,
            answer=answer,
            failure_reason=failure_reason,
        ),
    )


def _load_partial_trace_or_failure(*, trace_path: Path, task_id: str, failure_reason: str) -> dict[str, Any]:
    if not trace_path.exists():
        return _failure_run_result_payload(task_id, failure_reason)

    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return _failure_run_result_payload(task_id, failure_reason)
    if not isinstance(payload, dict):
        return _failure_run_result_payload(task_id, failure_reason)

    steps = payload.get("steps")
    if not isinstance(steps, list):
        steps = []
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        answer = None

    return _build_partial_run_payload(
        task_id=task_id,
        steps=steps,
        answer=answer,
        failure_reason=failure_reason,
    )


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _extract_plan_steps_from_payload(plan_solve: object) -> list[str]:
    plan_steps: list[str] = []
    if not isinstance(plan_solve, dict):
        return plan_steps

    raw_plan = plan_solve.get("plan", [])
    if not isinstance(raw_plan, list):
        return plan_steps

    for step in raw_plan:
        if isinstance(step, dict):
            raw_step_text = step.get("step")
            if isinstance(raw_step_text, str) and raw_step_text.strip():
                plan_steps.append(raw_step_text.strip())
            continue
        normalized_step = str(step).strip()
        if normalized_step:
            plan_steps.append(normalized_step)
    return plan_steps


def _build_short_term_memory_payload(
    *,
    task_question: str,
    run_result: dict[str, Any],
) -> dict[str, Any]:
    plan_solve = run_result.get("plan_solve")
    plan_steps = _extract_plan_steps_from_payload(plan_solve)
    answer_payload = run_result.get("answer")
    answer = dict(answer_payload) if isinstance(answer_payload, dict) else None

    raw_steps = run_result.get("steps", [])
    steps = raw_steps if isinstance(raw_steps, list) else []
    if not steps:
        existing_snapshot = run_result.get("short_term_memory")
        if isinstance(existing_snapshot, dict):
            snapshot = dict(existing_snapshot)
            snapshot.pop("status", None)
            snapshot.pop("recent_steps", None)
            snapshot["plan_steps"] = list(plan_steps)
            snapshot.setdefault("confirmed_evidence", [])
            snapshot.setdefault("recent_blockers", [])
            snapshot["answer"] = answer
            return snapshot

    memory = TaskShortTermMemory.replay(
        task_question=task_question,
        steps=steps,
        plan_steps=plan_steps,
        max_recent_steps=4,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason") if isinstance(run_result.get("failure_reason"), str) else None,
    )
    payload = memory.to_dict()
    payload["answer"] = answer
    return payload


def _tokenize_memory_text(text: str) -> set[str]:
    return {token.lower() for token in _MEMORY_TOKEN_RE.findall(text or "") if token.strip()}


def _memory_items_are_similar(left: MemoryItem, right: MemoryItem) -> bool:
    left_lesson = left.lesson.strip()
    right_lesson = right.lesson.strip()
    if not left_lesson or not right_lesson:
        return False

    normalized_left = " ".join(left_lesson.lower().split())
    normalized_right = " ".join(right_lesson.lower().split())
    if normalized_left == normalized_right:
        return True
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return True

    left_tokens = _tokenize_memory_text(left_lesson)
    right_tokens = _tokenize_memory_text(right_lesson)
    if not left_tokens or not right_tokens:
        return False

    overlap = left_tokens & right_tokens
    if not overlap:
        return False

    overlap_ratio = len(overlap) / min(len(left_tokens), len(right_tokens))
    jaccard = len(overlap) / len(left_tokens | right_tokens)
    return overlap_ratio >= 0.8 or (jaccard >= 0.7 and len(overlap) >= 5)


def _append_unique_memory_items(
    *,
    memory_store: LongTermMemory,
    items: list[MemoryItem],
) -> list[MemoryItem]:
    existing_items = memory_store.load_items()
    accepted_items: list[MemoryItem] = []

    for candidate in items:
        if not candidate.lesson.strip():
            continue
        if any(_memory_items_are_similar(candidate, existing_item) for existing_item in existing_items):
            continue
        if any(_memory_items_are_similar(candidate, accepted_item) for accepted_item in accepted_items):
            continue
        accepted_items.append(candidate)

    if accepted_items:
        memory_store.append_items(accepted_items)
    return accepted_items


def _write_short_term_memory_output(
    *,
    task_id: str,
    task_question: str,
    run_id: str,
    run_result: dict[str, Any],
    stm_payload: dict[str, Any] | None = None,
    output_root: Path = STM_OUTPUT_ROOT,
) -> Path:
    effective_payload = dict(stm_payload) if isinstance(stm_payload, dict) else _build_short_term_memory_payload(
        task_question=task_question,
        run_result=run_result,
    )
    stored_payload = dict(effective_payload)
    stm_output_dir = output_root / run_id
    stm_output_dir.mkdir(parents=True, exist_ok=True)
    stm_path = stm_output_dir / f"{task_id}.json"
    _write_json(stm_path, stored_payload)
    return stm_path


def _build_psrr_agent(*, config: AppConfig, model, tools: ToolRegistry, step_callback=None) -> PlanSolveReActReflexionAgent:
    return PlanSolveReActReflexionAgent(
        model=model,
        tools=tools,
        project_root=PROJECT_ROOT,
        step_callback=step_callback,
        config=PSRRAgentConfig(
            max_steps=config.agent.max_steps,
            memory_path=config.agent.memory_path,
            memory_use_vector_search=config.agent.memory_use_vector_search,
            memory_top_k=config.agent.memory_top_k,
            memory_max_items=config.agent.memory_max_items,
        ),
    )


def _build_psrr_react_sc_agent(*, config: AppConfig, model, tools: ToolRegistry, step_callback=None) -> PlanSolveReActSelfConsistencyAgent:
    return PlanSolveReActSelfConsistencyAgent(
        model=model,
        tools=tools,
        project_root=PROJECT_ROOT,
        step_callback=step_callback,
        config=PSRRSelfConsistencyAgentConfig(
            max_steps=config.agent.max_steps,
            memory_path=config.agent.memory_path,
            memory_use_vector_search=config.agent.memory_use_vector_search,
            memory_top_k=config.agent.memory_top_k,
            memory_max_items=config.agent.memory_max_items,
            branch_candidates=config.agent.branch_candidates,
            branch_target_tools=config.agent.branch_target_tools,
        ),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _build_improvement_memory_items(
    *,
    task_id: str,
    succeeded: bool,
    improvements: list[str],
) -> list[MemoryItem]:
    items: list[MemoryItem] = []
    seen_lessons: set[str] = set()
    for improvement in improvements[:8]:
        cleaned = improvement.strip()
        if not cleaned or cleaned in seen_lessons:
            continue
        seen_lessons.add(cleaned)
        items.append(
            MemoryItem(
                created_at=_now_iso(),
                tags=["improvement", "reflexion"],
                lesson=cleaned[:800],
                source_task_id=task_id,
                succeeded=succeeded,
            )
        )
    return items


def _build_retry_guidance(reflexion: ReflexionResult) -> list[str]:
    guidance = [item.strip() for item in reflexion.improvements if item.strip()]
    if guidance:
        return guidance[:8]

    memory_guidance = [item.lesson.strip() for item in reflexion.memory_items if item.lesson.strip()]
    if memory_guidance:
        return memory_guidance[:5]

    reflection = reflexion.reflection.strip()
    if reflection:
        return [reflection[:800]]
    return []


def _build_psrr_run_payload(
    *,
    run_result,
    psrr_agent,
    reflexion: ReflexionResult | None,
) -> dict[str, Any]:
    run_payload = run_result.to_dict()
    plan_steps = psrr_agent.last_plan.plan if psrr_agent.last_plan is not None else []
    focus_tools = psrr_agent.last_plan.focus_tools if psrr_agent.last_plan is not None else []
    task_analysis_payload = None
    if psrr_agent.last_task_analysis is not None:
        task_analysis_payload = {
            "intent": psrr_agent.last_task_analysis.intent,
            "subtasks": list(psrr_agent.last_task_analysis.subtasks),
        }

    structured_plan = [
        {
            "step": step,
            "tool": focus_tools[index] if index < len(focus_tools) else None,
        }
        for index, step in enumerate(plan_steps)
    ]
    plan_solve_payload = {
        "task_analysis": task_analysis_payload,
        "plan": structured_plan,
        "risks": (psrr_agent.last_plan.risks if psrr_agent.last_plan else []),
        "improvement_hints": list(psrr_agent.last_improvement_hints),
    }
    compact_payload = {
        "task_id": run_payload["task_id"],
        "answer": run_payload["answer"],
        "plan_solve": plan_solve_payload,
        "steps": run_payload["steps"],
        "failure_reason": run_payload["failure_reason"],
        "succeeded": run_payload["succeeded"],
    }
    if reflexion is not None:
        compact_payload["reflexion"] = reflexion.to_dict()
    return compact_payload


def _attempt_has_answer(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("answer"), dict)


def _select_preferred_attempt_payload(attempt_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempt_payloads:
        raise ValueError("attempt_payloads must not be empty.")

    successful_attempts = [payload for payload in attempt_payloads if bool(payload.get("succeeded"))]
    if successful_attempts:
        return dict(successful_attempts[-1])

    answered_attempts = [payload for payload in attempt_payloads if _attempt_has_answer(payload)]
    if answered_attempts:
        return dict(answered_attempts[-1])

    return dict(attempt_payloads[-1])


def _write_psrr_checkpoint(
    *,
    trace_path: Path | None,
    attempt_payloads: list[dict[str, Any]],
    current_payload: dict[str, Any] | None = None,
) -> None:
    if trace_path is None:
        return

    payloads = [dict(payload) for payload in attempt_payloads]
    if current_payload is not None:
        payloads.append(dict(current_payload))
    if not payloads:
        return

    checkpoint_payload = _select_preferred_attempt_payload(payloads)
    checkpoint_payload["attempt_count"] = len(payloads)
    if len(payloads) > 1:
        checkpoint_payload["attempts"] = payloads
    _write_json(trace_path, checkpoint_payload)


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    partial_trace_path: Path | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    effective_model = model or build_model_adapter(config)
    effective_tools = tools or create_default_tool_registry()

    if partial_trace_path is not None:
        _write_partial_trace(trace_path=partial_trace_path, task_id=task_id, steps=[])

        def step_callback(_step, state) -> None:
            _write_partial_trace(
                trace_path=partial_trace_path,
                task_id=task_id,
                steps=[step.to_dict() for step in state.steps],
                answer=state.answer.to_dict() if state.answer is not None else None,
                failure_reason=state.failure_reason,
            )
    else:
        step_callback = None

    agent_mode = (config.agent.mode or "react").strip().lower()

    psrr_agent: PlanSolveReActReflexionAgent | PlanSolveReActSelfConsistencyAgent | None = None
    if agent_mode in {"psrr", "psrr_react_sc"}:
        max_retry_count = 0
        if config.agent.enable_reflexion and (
            config.agent.reflexion_retry_on_failure or config.agent.reflexion_retry_always
        ):
            max_retry_count = max(0, config.agent.reflexion_max_retries)

        attempt_payloads: list[dict[str, Any]] = []
        retry_guidance: list[str] | None = None

        for attempt_index in range(max_retry_count + 1):
            if agent_mode == "psrr_react_sc":
                psrr_agent = _build_psrr_react_sc_agent(
                    config=config,
                    model=effective_model,
                    tools=effective_tools,
                    step_callback=step_callback,
                )
            else:
                psrr_agent = _build_psrr_agent(
                    config=config,
                    model=effective_model,
                    tools=effective_tools,
                    step_callback=step_callback,
                )
            run_result = psrr_agent.run(task, improvement_hints=retry_guidance)

            provisional_attempt_payload = _build_psrr_run_payload(
                run_result=run_result,
                psrr_agent=psrr_agent,
                reflexion=None,
            )
            provisional_attempt_payload["attempt_index"] = attempt_index + 1
            _write_psrr_checkpoint(
                trace_path=partial_trace_path,
                attempt_payloads=attempt_payloads,
                current_payload=provisional_attempt_payload,
            )

            reflexion: ReflexionResult | None = None
            if config.agent.enable_reflexion:
                plan = psrr_agent.last_plan.plan if psrr_agent.last_plan is not None else None
                reflexion = generate_reflexion(
                    model=effective_model,
                    task=task,
                    run_result=run_result,
                    plan=plan,
                )
                _append_unique_memory_items(
                    memory_store=psrr_agent.memory,
                    items=[
                        *reflexion.memory_items,
                        *_build_improvement_memory_items(
                            task_id=task.task_id,
                            succeeded=run_result.succeeded,
                            improvements=reflexion.improvements,
                        ),
                    ],
                )

            attempt_payload = _build_psrr_run_payload(
                run_result=run_result,
                psrr_agent=psrr_agent,
                reflexion=reflexion,
            )
            attempt_payload["attempt_index"] = attempt_index + 1
            attempt_payloads.append(attempt_payload)
            _write_psrr_checkpoint(
                trace_path=partial_trace_path,
                attempt_payloads=attempt_payloads,
            )

            if attempt_index >= max_retry_count or reflexion is None:
                break

            if run_result.succeeded and not config.agent.reflexion_retry_always:
                break

            retry_guidance = _build_retry_guidance(reflexion)

        if not attempt_payloads:
            raise RuntimeError("PSRR run did not produce an attempt payload.")

        final_payload = _select_preferred_attempt_payload(attempt_payloads)

        response_payload = {
            **final_payload,
            "attempt_count": len(attempt_payloads),
        }
        if len(attempt_payloads) > 1:
            response_payload["attempts"] = attempt_payloads
        return response_payload
    else:
        if agent_mode == "react_sc":
            agent = ReActSelfConsistencyAgent(
                model=effective_model,
                tools=effective_tools,
                config=ReActSelfConsistencyConfig(
                    max_steps=config.agent.max_steps,
                    branch_candidates=config.agent.branch_candidates,
                    branch_target_tools=config.agent.branch_target_tools,
                ),
                step_callback=step_callback,
            )
        else:
            agent = ReActAgent(
                model=effective_model,
                tools=effective_tools,
                config=ReActAgentConfig(max_steps=config.agent.max_steps),
                step_callback=step_callback,
            )
        run_result = agent.run(task)

    run_payload = run_result.to_dict()

    return run_payload


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    queue: multiprocessing.Queue[Any],
    partial_trace_path: Path | None,
) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(
                    task_id=task_id,
                    config=config,
                    partial_trace_path=partial_trace_path,
                ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig, partial_trace_path: Path | None = None) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config, partial_trace_path=partial_trace_path)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue, partial_trace_path),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _load_partial_trace_or_failure(
            trace_path=partial_trace_path,
            task_id=task_id,
            failure_reason=f"Task timed out after {timeout_seconds} seconds.",
        ) if partial_trace_path is not None else _failure_run_result_payload(
            task_id,
            f"Task timed out after {timeout_seconds} seconds.",
        )

    if queue.empty():
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    result = queue.get()
    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


def _write_task_outputs(
    task_id: str,
    run_output_dir: Path,
    run_result: dict[str, Any],
    *,
    task_question: str,
) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)
    stm_payload = _build_short_term_memory_payload(
        task_question=task_question,
        run_result=run_result,
    )
    _write_short_term_memory_output(
        task_id=task_id,
        task_question=task_question,
        run_id=run_output_dir.name,
        run_result=run_result,
        stm_payload=stm_payload,
    )

    reflexion_payload = run_result.get("reflexion")
    if isinstance(reflexion_payload, dict):
        reflexion_path = task_output_dir / "reflexion.json"
        _write_json(reflexion_path, reflexion_payload)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    task_question = DABenchPublicDataset(config.dataset.root_path).get_task(task_id).question
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    partial_trace_path = task_output_dir / "trace.json"
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            partial_trace_path=partial_trace_path,
        )
    else:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            model=model,
            tools=tools,
            partial_trace_path=partial_trace_path,
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(
        task_id,
        run_output_dir,
        run_result,
        task_question=task_question,
    )


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
