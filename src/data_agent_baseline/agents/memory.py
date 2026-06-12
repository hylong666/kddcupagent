from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", flags=re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if token.strip()}


@dataclass(frozen=True, slots=True)
class MemoryItem:
    created_at: str
    tags: list[str]
    lesson: str
    source_task_id: str | None = None
    succeeded: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "tags": list(self.tags),
            "lesson": self.lesson,
            "source_task_id": self.source_task_id,
            "succeeded": self.succeeded,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryItem:
        created_at = str(payload.get("created_at") or "")
        tags = payload.get("tags")
        lesson = payload.get("lesson")
        if not created_at:
            created_at = _now_iso()
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            tags = []
        if not isinstance(lesson, str):
            lesson = ""
        source_task_id = payload.get("source_task_id")
        succeeded = payload.get("succeeded")
        return cls(
            created_at=created_at,
            tags=[tag.strip() for tag in tags if tag.strip()],
            lesson=lesson.strip(),
            source_task_id=str(source_task_id) if isinstance(source_task_id, str) and source_task_id else None,
            succeeded=bool(succeeded) if succeeded is not None else None,
        )


class LongTermMemory:
    def __init__(self, *, path: Path, max_items: int = 2000) -> None:
        self.path = path
        self.max_items = max(1, int(max_items))

    def load_items(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []

        items: list[MemoryItem] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            item = MemoryItem.from_dict(payload)
            if not item.lesson:
                continue
            items.append(item)

        if len(items) > self.max_items:
            items = items[-self.max_items :]
        return items

    def append_items(self, items: Iterable[MemoryItem]) -> None:
        normalized = [item for item in items if item.lesson]
        if not normalized:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        serialized_lines = []
        for item in normalized:
            serialized_lines.append(json.dumps(item.to_dict(), ensure_ascii=False))
        text = "\n".join(serialized_lines) + "\n"

        try:
            import fcntl  # type: ignore

            with self.path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(text)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(text)

    def retrieve(self, *, query: str, k: int = 3) -> list[MemoryItem]:
        candidates = self.load_items()
        if not candidates:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return candidates[-k:]

        def score(item: MemoryItem) -> int:
            tag_tokens = _tokenize(" ".join(item.tags))
            lesson_tokens = _tokenize(item.lesson)
            overlap = len(query_tokens & (tag_tokens | lesson_tokens))
            return overlap

        ranked = sorted(candidates, key=score, reverse=True)
        picked: list[MemoryItem] = []
        for item in ranked:
            if len(picked) >= k:
                break
            if score(item) <= 0:
                break
            picked.append(item)

        if not picked:
            picked = candidates[-k:]
        return picked


def default_memory_path(project_root: Path) -> Path:
    return project_root / "artifacts" / "memory" / "long_term_memory.jsonl"
