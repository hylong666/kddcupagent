from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    mode: str = "react"
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
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

    enable_reflexion: bool = True
    memory_path: Path | None = None
    memory_use_vector_search: bool = False
    memory_top_k: int = 3
    memory_max_items: int = 2000
    reflexion_retry_on_failure: bool = True
    reflexion_retry_always: bool = False
    reflexion_max_retries: int = 1


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _optional_path_value(raw_value: str | None) -> Path | None:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    if not normalized:
        return None
    return _path_value(normalized, PROJECT_ROOT)


def _string_tuple_value(raw_value: object, default_value: tuple[str, ...]) -> tuple[str, ...]:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        return (normalized,) if normalized else default_value
    if isinstance(raw_value, (list, tuple)):
        normalized_items = tuple(str(item).strip() for item in raw_value if str(item).strip())
        return normalized_items or default_value
    return default_value


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        mode=str(agent_payload.get("mode", agent_defaults.mode)),
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        branch_candidates=int(agent_payload.get("branch_candidates", agent_defaults.branch_candidates)),
        branch_target_tools=_string_tuple_value(
            agent_payload.get("branch_target_tools"),
            agent_defaults.branch_target_tools,
        ),
        enable_reflexion=bool(agent_payload.get("enable_reflexion", agent_defaults.enable_reflexion)),
        memory_path=_optional_path_value(agent_payload.get("memory_path")),
        memory_use_vector_search=bool(
            agent_payload.get("memory_use_vector_search", agent_defaults.memory_use_vector_search)
        ),
        memory_top_k=int(agent_payload.get("memory_top_k", agent_defaults.memory_top_k)),
        memory_max_items=int(agent_payload.get("memory_max_items", agent_defaults.memory_max_items)),
        reflexion_retry_on_failure=bool(
            agent_payload.get("reflexion_retry_on_failure", agent_defaults.reflexion_retry_on_failure)
        ),
        reflexion_retry_always=bool(
            agent_payload.get("reflexion_retry_always", agent_defaults.reflexion_retry_always)
        ),
        reflexion_max_retries=int(agent_payload.get("reflexion_max_retries", agent_defaults.reflexion_max_retries)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
