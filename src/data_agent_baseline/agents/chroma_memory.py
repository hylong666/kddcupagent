from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Iterable

from data_agent_baseline.agents.memory import LongTermMemory, MemoryItem


def _collection_name_for_path(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return f"ltm-{digest[:16]}"


def _default_index_path(path: Path) -> Path:
    return path.parent / f"{path.stem}_chroma"


def _create_chroma_client(path: Path) -> Any:
    chromadb = importlib.import_module("chromadb")
    return chromadb.PersistentClient(path=str(path))


def _create_embedding_function() -> Any:
    embedding_functions = importlib.import_module("chromadb.utils.embedding_functions")
    return embedding_functions.DefaultEmbeddingFunction()


class ChromaLongTermMemory(LongTermMemory):
    def __init__(self, *, path: Path, max_items: int = 2000) -> None:
        super().__init__(path=path, max_items=max_items)
        self.index_path = _default_index_path(path)
        self.collection_name = _collection_name_for_path(path)
        self._client: Any | None = None
        self._collection: Any | None = None

    def append_items(self, items: Iterable[MemoryItem]) -> None:
        normalized = [item for item in items if item.lesson]
        if not normalized:
            return

        try:
            self._get_collection().add(
                ids=self._memory_item_ids(normalized),
                documents=[self._item_document(item) for item in normalized],
                metadatas=[self._item_metadata(item) for item in normalized],
            )
            self._trim_to_max_items()
        except Exception:
            self._collection = None
            raise

    def load_items(self) -> list[MemoryItem]:
        try:
            payload = self._get_collection().get(include=["metadatas"])
        except Exception:
            return []

        raw_metadatas = payload.get("metadatas") or []
        items: list[MemoryItem] = []
        for metadata in raw_metadatas:
            if not isinstance(metadata, dict):
                continue
            item = self._item_from_metadata(metadata)
            if item is None:
                continue
            items.append(item)

        items.sort(key=lambda item: item.created_at)
        if len(items) > self.max_items:
            items = items[-self.max_items :]
        return items

    def retrieve(self, *, query: str, k: int = 3) -> list[MemoryItem]:
        normalized_query = query.strip()
        if not normalized_query:
            return self._latest_items(k)

        try:
            result = self._get_collection().query(
                query_texts=[normalized_query],
                n_results=max(1, int(k)),
                include=["metadatas"],
            )
        except Exception:
            return self._latest_items(k)

        raw_metadatas = result.get("metadatas") or []
        metadatas = raw_metadatas[0] if raw_metadatas else []
        picked: list[MemoryItem] = []
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            item = self._item_from_metadata(metadata)
            if item is None:
                continue
            picked.append(item)
            if len(picked) >= k:
                break

        # Trust Chroma's semantic ranking first. If it returns nothing usable,
        # fall back to the latest k memory items to preserve the old behavior.
        if not picked:
            return self._latest_items(k)
        return picked

    def _item_document(self, item: MemoryItem) -> str:
        tags = ", ".join(tag for tag in item.tags if tag.strip())
        if tags:
            return f"Tags: {tags}\nLesson: {item.lesson}"
        return item.lesson

    def _item_metadata(self, item: MemoryItem) -> dict[str, str | bool]:
        metadata: dict[str, str | bool] = {
            "created_at": item.created_at,
            "lesson": item.lesson,
            "tags_json": json.dumps(item.tags, ensure_ascii=False),
        }
        if item.source_task_id is not None:
            metadata["source_task_id"] = item.source_task_id
        if item.succeeded is not None:
            metadata["succeeded"] = item.succeeded
        return metadata

    def _item_from_metadata(self, metadata: dict[str, Any]) -> MemoryItem | None:
        lesson = metadata.get("lesson")
        created_at = metadata.get("created_at")
        tags_json = metadata.get("tags_json")
        if not isinstance(lesson, str) or not lesson.strip():
            return None
        if not isinstance(created_at, str) or not created_at.strip():
            created_at = ""
        tags: list[str] = []
        if isinstance(tags_json, str) and tags_json.strip():
            try:
                parsed_tags = json.loads(tags_json)
            except json.JSONDecodeError:
                parsed_tags = []
            if isinstance(parsed_tags, list):
                tags = [str(tag).strip() for tag in parsed_tags if str(tag).strip()]

        source_task_id = metadata.get("source_task_id")
        succeeded = metadata.get("succeeded")
        return MemoryItem(
            created_at=created_at,
            tags=tags,
            lesson=lesson.strip(),
            source_task_id=source_task_id if isinstance(source_task_id, str) and source_task_id else None,
            succeeded=bool(succeeded) if isinstance(succeeded, bool) else None,
        )

    def _memory_item_ids(self, items: list[MemoryItem]) -> list[str]:
        ids: list[str] = []
        for item in items:
            payload = json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)
            ids.append(hashlib.sha1(payload.encode("utf-8")).hexdigest())
        return ids

    def _trim_to_max_items(self) -> None:
        payload = self._get_collection().get(include=["metadatas"])
        raw_ids = payload.get("ids") or []
        raw_metadatas = payload.get("metadatas") or []
        paired_items: list[tuple[str, MemoryItem]] = []
        for item_id, metadata in zip(raw_ids, raw_metadatas, strict=False):
            if not isinstance(item_id, str) or not isinstance(metadata, dict):
                continue
            item = self._item_from_metadata(metadata)
            if item is None:
                continue
            paired_items.append((item_id, item))

        if len(paired_items) <= self.max_items:
            return

        paired_items.sort(key=lambda pair: pair[1].created_at)
        ids_to_remove = [item_id for item_id, _ in paired_items[: len(paired_items) - self.max_items]]
        if ids_to_remove:
            self._get_collection().delete(ids=ids_to_remove)

    def _latest_items(self, k: int) -> list[MemoryItem]:
        items = self.load_items()
        if not items:
            return []
        return items[-k:]

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if self._client is None:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._client = _create_chroma_client(self.index_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=_create_embedding_function(),
        )
        return self._collection
