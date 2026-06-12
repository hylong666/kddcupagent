from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.agents.chroma_memory import ChromaLongTermMemory
from data_agent_baseline.agents.memory import LongTermMemory, default_memory_path
from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.plan_solve import (
    PlanSolveResult,
    TaskAnalysisResult,
    generate_plan_solve,
    generate_task_analysis,
    render_memory_snippets,
)
from data_agent_baseline.agents.prompt import PSRR_REACT_SYSTEM_PROMPT, build_task_prompt
from data_agent_baseline.agents.react_sc import ReActSelfConsistencyAgent, ReActSelfConsistencyConfig
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.agents.short_term_memory import TaskShortTermMemory
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class PSRRSelfConsistencyAgentConfig:
    max_steps: int = 16
    memory_path: Path | None = None
    memory_use_vector_search: bool = False
    memory_top_k: int = 3
    memory_max_items: int = 2000
    branch_candidates: int = 3
    branch_target_tools: tuple[str, ...] = (
        "execute_python",
        "read_csv",
        "read_json",
        "read_doc",
        "inspect_sqlite_schema",
        "execute_context_sql",
    )


class PlanSolveReActSelfConsistencyAgent:
    """Plan-Solve + Self-Consistency ReAct agent wrapper."""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        project_root: Path,
        config: PSRRSelfConsistencyAgentConfig | None = None,
        step_callback=None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.project_root = project_root
        self.config = config or PSRRSelfConsistencyAgentConfig()
        self.step_callback = step_callback

        memory_path = self.config.memory_path or default_memory_path(project_root)
        memory_cls = ChromaLongTermMemory if self.config.memory_use_vector_search else LongTermMemory
        self.memory = memory_cls(path=memory_path, max_items=self.config.memory_max_items)

        self.last_plan: PlanSolveResult | None = None
        self.last_task_analysis: TaskAnalysisResult | None = None
        self.last_memory_snippets: str | None = None
        self.last_improvement_hints: list[str] = []
        self.last_short_term_memory_snapshot: dict[str, object] | None = None

    def _build_psrr_system_prompt(
        self,
        plan: PlanSolveResult,
    ) -> str:
        plan_lines = "\n".join(f"- {step}" for step in plan.plan)
        plan_block = (
            "\n\nPlan-Solve plan (execute strictly in this order):\n"
            f"{plan_lines}\n"
            "Do not skip, reorder, or revise plan steps during execution."
        )

        risk_block = ""
        if plan.risks:
            rendered_risks = "\n".join(f"- {item}" for item in plan.risks if item.strip())
            if rendered_risks:
                risk_block = (
                    "\n\nProactive reflection: likely failure modes or pitfalls to guard against in this attempt:\n"
                    f"{rendered_risks}\n"
                )

        sc_block = (
            "\n\nCritical-step interpretation policy:\n"
            "- For designated read or inspection tools, generate 2-3 distinct candidate actions.\n"
            "- Execute the candidates, compare observations, and keep the strongest branch.\n"
            "- Prefer branches that succeed and return non-empty evidence.\n"
        )

        return (PSRR_REACT_SYSTEM_PROMPT.strip() + plan_block + risk_block + sc_block).strip()

    def run(self, task: PublicTask, *, improvement_hints: list[str] | None = None) -> AgentRunResult:
        short_term_memory = TaskShortTermMemory.from_task(task)

        retrieved_memory_items = []
        try:
            retrieved_memory_items = self.memory.retrieve(query=task.question, k=self.config.memory_top_k)
        except Exception:
            retrieved_memory_items = []

        snippet_items = []
        retrieved_improvement_hints: list[str] = []
        for item in retrieved_memory_items:
            tags = {tag.strip().lower() for tag in item.tags if tag.strip()}
            normalized = item.lesson.strip()
            if not normalized:
                continue
            if "improvement" in tags or "reflexion" in tags:
                retrieved_improvement_hints.append(normalized)
                continue
            snippet_items.append(item)

        rendered = render_memory_snippets(snippet_items)
        memory_snippets: str | None = rendered or None
        self.last_memory_snippets = memory_snippets
        merged_improvement_hints: list[str] = []
        seen_hints: set[str] = set()

        for item in improvement_hints or []:
            normalized = item.strip()
            if not normalized or normalized in seen_hints:
                continue
            merged_improvement_hints.append(normalized)
            seen_hints.add(normalized)

        for normalized in retrieved_improvement_hints:
            if not normalized or normalized in seen_hints:
                continue
            merged_improvement_hints.append(normalized)
            seen_hints.add(normalized)
        self.last_improvement_hints = merged_improvement_hints

        task_analysis: TaskAnalysisResult | None = None
        try:
            task_analysis = generate_task_analysis(
                model=self.model,
                task=task,
                tool_descriptions=self.tools.describe_for_prompt(),
                memory_snippets=memory_snippets,
            )
        except Exception:
            task_analysis = None
        self.last_task_analysis = task_analysis

        try:
            plan = generate_plan_solve(
                model=self.model,
                task=task,
                tool_descriptions=self.tools.describe_for_prompt(),
                task_analysis=task_analysis,
                improvement_hints=merged_improvement_hints,
            )
        except Exception as exc:
            plan = PlanSolveResult(
                plan=[
                    "List the context directory to see available files.",
                    "Inspect relevant docs, JSON, CSV, or schemas to locate the needed data.",
                    "For key reading or inspection steps, compare multiple candidate actions before continuing.",
                    "Use the strongest observation to narrow the answer table.",
                    "Submit the final answer table via the answer tool.",
                ],
                focus_tools=[],
                risks=[f"planning_failed: {exc}"],
                raw_response="",
            )
        self.last_plan = plan
        short_term_memory.set_plan(plan.plan)

        def on_step(step, state) -> None:
            short_term_memory.record_step(step, state)
            if self.step_callback is not None:
                self.step_callback(step, state)

        react_agent = ReActSelfConsistencyAgent(
            model=self.model,
            tools=self.tools,
            config=ReActSelfConsistencyConfig(
                max_steps=self.config.max_steps,
                branch_candidates=self.config.branch_candidates,
                branch_target_tools=self.config.branch_target_tools,
            ),
            system_prompt=self._build_psrr_system_prompt(plan),
            task_prompt_builder=lambda current_task, _state: build_task_prompt(
                current_task,
                extra_context=short_term_memory.build_execution_context(),
                include_question=False,
            ),
            step_callback=on_step,
        )
        run_result = react_agent.run(task)
        short_term_memory.finalize(succeeded=run_result.succeeded, failure_reason=run_result.failure_reason)
        self.last_short_term_memory_snapshot = short_term_memory.to_dict()
        return run_result
