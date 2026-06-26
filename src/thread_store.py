from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

try:
    from .loop_engine import (
        DEFAULT_LOOP_RECIPE_ID,
        LOOP_RECIPE_SCHEMA_VERSION,
        LoopRecipe,
        default_loop_recipe,
    )
except ImportError:
    from loop_engine import (
        DEFAULT_LOOP_RECIPE_ID,
        LOOP_RECIPE_SCHEMA_VERSION,
        LoopRecipe,
        default_loop_recipe,
    )


DEFAULT_THREAD_TITLE = "New thread"
MAX_THREAD_TITLE_LENGTH = 64
MAX_LOOP_RECIPE_LIST_LIMIT = 50
MAX_LOOP_RUN_LIST_LIMIT = 50


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def default_thread_id() -> str:
    return f"thread_{uuid.uuid4()}"


def default_thread_instance_id() -> str:
    return uuid.uuid4().hex


def default_recipe_id() -> str:
    return f"recipe_{uuid.uuid4().hex}"


def safe_title(value: str | None) -> str:
    title = " ".join(str(value or "").split()).strip()
    if not title:
        return DEFAULT_THREAD_TITLE
    return title[:MAX_THREAD_TITLE_LENGTH]


def safe_record_id(value: str | None, *, field_name: str) -> str:
    record_id = " ".join(str(value or "").split()).strip()
    if not record_id:
        raise ValueError(f"{field_name} must not be empty.")
    if len(record_id.encode("utf-8")) > 128:
        raise ValueError(f"{field_name} is too long.")
    if not all(char.isalnum() or char in {"_", "-", ".", ":"} for char in record_id):
        raise ValueError(f"{field_name} contains unsupported characters.")
    return record_id


def json_dumps(value: dict | None) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_loads(value: str | None) -> Optional[dict]:
    if not value:
        return None
    data = json.loads(value)
    return data if isinstance(data, dict) else None


def required_json_loads(value: str | None) -> dict:
    data = json_loads(value)
    if data is None:
        raise ValueError("Stored JSON payload must be an object.")
    return data


def json_list_dumps(value: tuple[str, ...] | list[str] | None) -> str:
    return json.dumps(list(value or ()), sort_keys=True, separators=(",", ":"))


def json_list_loads(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    data = json.loads(value)
    if not isinstance(data, list):
        return ()
    return tuple(str(item) for item in data)


def loop_run_metadata_from_report(report: Mapping) -> dict:
    run = report.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("Loop report must contain a run object.")
    run_id = safe_record_id(str(run.get("run_id") or ""), field_name="run_id")
    steps = run.get("steps") or ()
    if not isinstance(steps, list):
        steps = []
    metadata = run.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "run_id": run_id,
        "schema_version": str(report.get("schema_version") or "loop-report/v1"),
        "final_decision": run.get("final_decision"),
        "context_provider": run.get("context_provider"),
        "backend": run.get("backend"),
        "model_label": run.get("model_label"),
        "recipe_id": metadata.get("recipe_id"),
        "recipe_name": metadata.get("recipe_name"),
        "step_count": len(steps),
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
    }


@dataclass(frozen=True)
class LoopRunRecord:
    run_id: str
    thread_id: str
    schema_version: str
    raw_report: dict
    public_report: dict
    final_decision: Optional[str]
    context_provider: Optional[str]
    backend: Optional[str]
    model_label: Optional[str]
    recipe_id: Optional[str]
    recipe_name: Optional[str]
    step_count: int
    started_at: Optional[str]
    completed_at: Optional[str]
    created_at: str
    user_message_id: Optional[int] = None
    assistant_message_id: Optional[int] = None

    def summary_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "final_decision": self.final_decision,
            "context_provider": self.context_provider,
            "backend": self.backend,
            "model": self.model_label,
            "recipe_id": self.recipe_id,
            "recipe_name": self.recipe_name,
            "step_count": self.step_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
        }

    def detail_dict(self, *, public: bool = True) -> dict:
        payload = self.summary_dict()
        payload["report"] = self.public_report if public else self.raw_report
        payload["public"] = public
        return payload


@dataclass(frozen=True)
class ThreadMessage:
    id: int
    thread_id: str
    role: str
    content: str
    created_at: str
    thinking: Optional[dict] = None
    loop_payload: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "thinking": self.thinking,
            "loop_payload": self.loop_payload,
        }


@dataclass(frozen=True)
class ThreadMemory:
    message_id: int
    thread_id: str
    role: str
    content: str
    created_at: str
    score: float

    def to_context_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "score": self.score,
        }


@dataclass(frozen=True)
class ThreadRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    instance_id: str
    generation: int = 0
    message_count: int = 0
    loop_run_count: int = 0
    messages: tuple[ThreadMessage, ...] = ()
    loop_runs: tuple[LoopRunRecord, ...] = ()
    latest: Optional[dict] = None

    def summary_dict(self, *, include_latest: bool = False) -> dict:
        payload = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "loop_run_count": self.loop_run_count,
        }
        if include_latest:
            payload["latest"] = self.latest
        return payload

    def detail_dict(self) -> dict:
        payload = self.summary_dict(include_latest=True)
        payload["messages"] = [message.to_dict() for message in self.messages]
        payload["loop_runs"] = [run.summary_dict() for run in self.loop_runs]
        return payload


class ThreadStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._create_schema()

    @classmethod
    def in_memory(cls) -> "ThreadStore":
        return cls(":memory:")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                instance_id TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                thinking_json TEXT,
                loop_payload_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_thread_id_id
                ON messages(thread_id, id);

            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY
                    REFERENCES messages(id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL
                    REFERENCES threads(id) ON DELETE CASCADE,
                embedding_model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_message_embeddings_thread_model
                ON message_embeddings(thread_id, embedding_model);

            CREATE TABLE IF NOT EXISTS loop_runs (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                user_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                assistant_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                schema_version TEXT NOT NULL,
                raw_report_json TEXT NOT NULL,
                public_report_json TEXT NOT NULL,
                final_decision TEXT,
                context_provider TEXT,
                backend TEXT,
                model_label TEXT,
                recipe_id TEXT,
                recipe_name TEXT,
                step_count INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_loop_runs_thread_created
                ON loop_runs(thread_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS loop_recipes (
                recipe_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                goal TEXT NOT NULL,
                instructions TEXT NOT NULL DEFAULT '',
                success_criteria_json TEXT NOT NULL DEFAULT '[]',
                stop_condition TEXT NOT NULL,
                context_provider TEXT NOT NULL DEFAULT 'auto',
                model_profile TEXT NOT NULL DEFAULT 'quality',
                verifier TEXT NOT NULL DEFAULT 'default',
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_loop_recipes_updated
                ON loop_recipes(updated_at DESC);
            """
        )
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(threads)").fetchall()
        }
        if "generation" not in columns:
            self._conn.execute(
                "ALTER TABLE threads ADD COLUMN generation INTEGER NOT NULL DEFAULT 0"
            )
        if "instance_id" not in columns:
            self._conn.execute(
                "ALTER TABLE threads ADD COLUMN instance_id TEXT NOT NULL DEFAULT ''"
            )
        for row in self._conn.execute(
            "SELECT id FROM threads WHERE instance_id = ''"
        ).fetchall():
            self._conn.execute(
                "UPDATE threads SET instance_id = ? WHERE id = ?",
                (default_thread_instance_id(), str(row["id"])),
            )
        self._conn.commit()

    def create_thread(
        self,
        *,
        title: str | None = None,
        thread_id: str | None = None,
    ) -> ThreadRecord:
        now = utc_now()
        record_id = thread_id or default_thread_id()
        instance_id = default_thread_instance_id()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO threads (
                    id, title, created_at, updated_at, instance_id
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (record_id, safe_title(title), now, now, instance_id),
            )
            self._conn.commit()
            record = self.get_thread(record_id)
            if record is None:
                raise RuntimeError("Thread was not created.")
            return record

    def ensure_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
    ) -> ThreadRecord:
        with self._lock:
            record = self.get_thread(thread_id)
            if record is not None:
                return record
            return self.create_thread(title=title, thread_id=thread_id)

    def list_threads(self, *, limit: int = 30) -> list[ThreadRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    t.id,
                    t.title,
                    t.created_at,
                    t.updated_at,
                    t.instance_id,
                    t.generation,
                    COUNT(m.id) AS message_count,
                    (
                        SELECT COUNT(*)
                        FROM loop_runs lr
                        WHERE lr.thread_id = t.id
                    ) AS loop_run_count,
                    (
                        SELECT loop_payload_json
                        FROM messages latest
                        WHERE latest.thread_id = t.id
                          AND latest.loop_payload_json IS NOT NULL
                        ORDER BY latest.id DESC
                        LIMIT 1
                    ) AS latest_json
                FROM threads t
                LEFT JOIN messages m ON m.thread_id = t.id
                GROUP BY
                    t.id, t.title, t.created_at, t.updated_at,
                    t.instance_id, t.generation
                ORDER BY t.updated_at DESC, t.created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
            return [self._thread_from_row(row, messages=()) for row in rows]

    def get_thread(self, thread_id: str) -> Optional[ThreadRecord]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    t.id,
                    t.title,
                    t.created_at,
                    t.updated_at,
                    t.instance_id,
                    t.generation,
                    COUNT(m.id) AS message_count,
                    (
                        SELECT COUNT(*)
                        FROM loop_runs lr
                        WHERE lr.thread_id = t.id
                    ) AS loop_run_count,
                    (
                        SELECT loop_payload_json
                        FROM messages latest
                        WHERE latest.thread_id = t.id
                          AND latest.loop_payload_json IS NOT NULL
                        ORDER BY latest.id DESC
                        LIMIT 1
                    ) AS latest_json
                FROM threads t
                LEFT JOIN messages m ON m.thread_id = t.id
                WHERE t.id = ?
                GROUP BY
                    t.id, t.title, t.created_at, t.updated_at,
                    t.instance_id, t.generation
                """,
                (thread_id,),
            ).fetchone()
            if row is None:
                return None
            messages = tuple(
                self._message_from_row(message_row)
                for message_row in self._conn.execute(
                    """
                    SELECT id, thread_id, role, content, thinking_json,
                           loop_payload_json, created_at
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY id ASC
                    """,
                    (thread_id,),
                ).fetchall()
            )
            loop_runs = tuple(
                self._loop_run_from_row(run_row)
                for run_row in self._conn.execute(
                    """
                    SELECT
                        run_id, thread_id, user_message_id, assistant_message_id,
                        schema_version, raw_report_json, public_report_json,
                        final_decision, context_provider, backend, model_label,
                        recipe_id, recipe_name, step_count, started_at,
                        completed_at, created_at
                    FROM loop_runs
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (thread_id, MAX_LOOP_RUN_LIST_LIMIT),
                ).fetchall()
            )
            return self._thread_from_row(row, messages=messages, loop_runs=loop_runs)

    def recent_messages(self, thread_id: str, limit: int = 12) -> tuple[ThreadMessage, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, thread_id, role, content, thinking_json,
                       loop_payload_json, created_at
                FROM messages
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, max(1, int(limit))),
            ).fetchall()
            return tuple(
                self._message_from_row(row) for row in reversed(rows)
            )

    def has_message_embeddings(
        self,
        thread_id: str,
        embedding_model: str,
    ) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1
                FROM message_embeddings
                WHERE thread_id = ? AND embedding_model = ?
                LIMIT 1
                """,
                (thread_id, str(embedding_model or "")),
            ).fetchone()
            return row is not None

    def upsert_message_embedding(
        self,
        message: ThreadMessage,
        *,
        embedding_model: str,
        vector: list[float] | tuple[float, ...],
    ) -> bool:
        vector_json = vector_to_json(vector)
        now = utc_now()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id
                FROM messages
                WHERE id = ? AND thread_id = ?
                """,
                (message.id, message.thread_id),
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                """
                INSERT OR REPLACE INTO message_embeddings (
                    message_id, thread_id, embedding_model, vector_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.thread_id,
                    str(embedding_model or ""),
                    vector_json,
                    now,
                ),
            )
            self._conn.commit()
            return True

    def semantic_memories(
        self,
        thread_id: str,
        *,
        embedding_model: str,
        query_vector: list[float] | tuple[float, ...],
        limit: int = 4,
        exclude_message_ids: tuple[int, ...] = (),
        min_score: float = 0.05,
    ) -> tuple[ThreadMemory, ...]:
        query = validate_vector(query_vector)
        excluded = {int(message_id) for message_id in exclude_message_ids}
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    m.id,
                    m.thread_id,
                    m.role,
                    m.content,
                    m.created_at,
                    e.vector_json
                FROM message_embeddings e
                JOIN messages m ON m.id = e.message_id
                WHERE e.thread_id = ? AND e.embedding_model = ?
                ORDER BY m.id ASC
                """,
                (thread_id, str(embedding_model or "")),
            ).fetchall()

        scored = []
        for row in rows:
            message_id = int(row["id"])
            if message_id in excluded:
                continue
            score = cosine_similarity(query, vector_from_json(row["vector_json"]))
            if score < float(min_score):
                continue
            scored.append(
                ThreadMemory(
                    message_id=message_id,
                    thread_id=str(row["thread_id"]),
                    role=str(row["role"]),
                    content=str(row["content"]),
                    created_at=str(row["created_at"]),
                    score=round(score, 6),
                )
            )

        scored.sort(key=lambda memory: (-memory.score, memory.message_id))
        return tuple(scored[: max(1, int(limit))])

    def list_loop_runs(
        self,
        thread_id: str,
        *,
        limit: int = MAX_LOOP_RUN_LIST_LIMIT,
    ) -> tuple[LoopRunRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    run_id, thread_id, user_message_id, assistant_message_id,
                    schema_version, raw_report_json, public_report_json,
                    final_decision, context_provider, backend, model_label,
                    recipe_id, recipe_name, step_count, started_at,
                    completed_at, created_at
                FROM loop_runs
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (thread_id, max(1, int(limit))),
            ).fetchall()
            return tuple(self._loop_run_from_row(row) for row in rows)

    def get_loop_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> Optional[LoopRunRecord]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    run_id, thread_id, user_message_id, assistant_message_id,
                    schema_version, raw_report_json, public_report_json,
                    final_decision, context_provider, backend, model_label,
                    recipe_id, recipe_name, step_count, started_at,
                    completed_at, created_at
                FROM loop_runs
                WHERE thread_id = ? AND run_id = ?
                """,
                (thread_id, run_id),
            ).fetchone()
            return self._loop_run_from_row(row) if row else None

    def ensure_default_recipe(self) -> LoopRecipe:
        with self._lock:
            existing = self.get_recipe(DEFAULT_LOOP_RECIPE_ID)
            if existing is not None:
                return existing
            recipe = default_loop_recipe()
            self._insert_recipe(recipe)
            self._conn.commit()
            return recipe

    def list_recipes(
        self,
        *,
        limit: int = MAX_LOOP_RECIPE_LIST_LIMIT,
    ) -> tuple[LoopRecipe, ...]:
        with self._lock:
            self.ensure_default_recipe()
            rows = self._conn.execute(
                """
                SELECT
                    recipe_id, schema_version, name, description, goal,
                    instructions, success_criteria_json, stop_condition,
                    context_provider, model_profile, verifier, metadata_json,
                    created_at, updated_at
                FROM loop_recipes
                ORDER BY
                    CASE WHEN recipe_id = ? THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT ?
                """,
                (DEFAULT_LOOP_RECIPE_ID, max(1, int(limit))),
            ).fetchall()
            return tuple(self._recipe_from_row(row) for row in rows)

    def get_recipe(self, recipe_id: str) -> Optional[LoopRecipe]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    recipe_id, schema_version, name, description, goal,
                    instructions, success_criteria_json, stop_condition,
                    context_provider, model_profile, verifier, metadata_json,
                    created_at, updated_at
                FROM loop_recipes
                WHERE recipe_id = ?
                """,
                (recipe_id,),
            ).fetchone()
            return self._recipe_from_row(row) if row else None

    def create_recipe(
        self,
        *,
        name: str,
        goal: str,
        instructions: str = "",
        success_criteria: tuple[str, ...] | list[str] = (),
        stop_condition: str = "",
        context_provider: str = "auto",
        model_profile: str = "quality",
        verifier: str = "default",
        description: str = "",
        metadata: Optional[Mapping] = None,
        recipe_id: Optional[str] = None,
    ) -> LoopRecipe:
        now = datetime.now(timezone.utc)
        recipe = LoopRecipe(
            recipe_id=safe_record_id(
                recipe_id or default_recipe_id(),
                field_name="recipe_id",
            ),
            name=name,
            description=description,
            goal=goal,
            instructions=instructions,
            success_criteria=tuple(success_criteria or ()),
            stop_condition=stop_condition or default_loop_recipe().stop_condition,
            context_provider=context_provider,
            model_profile=model_profile,
            verifier=verifier,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        with self._lock:
            self._insert_recipe(recipe)
            self._conn.commit()
            return recipe

    def update_recipe(
        self,
        recipe_id: str,
        *,
        name: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[str] = None,
        success_criteria: Optional[tuple[str, ...] | list[str]] = None,
        stop_condition: Optional[str] = None,
        context_provider: Optional[str] = None,
        model_profile: Optional[str] = None,
        verifier: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping] = None,
    ) -> LoopRecipe:
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = self.get_recipe(recipe_id)
            if existing is None:
                raise KeyError(recipe_id)
            recipe = LoopRecipe(
                recipe_id=existing.recipe_id,
                name=existing.name if name is None else name,
                description=(
                    existing.description if description is None else description
                ),
                goal=existing.goal if goal is None else goal,
                instructions=(
                    existing.instructions
                    if instructions is None
                    else instructions
                ),
                success_criteria=(
                    existing.success_criteria
                    if success_criteria is None
                    else tuple(success_criteria)
                ),
                stop_condition=(
                    existing.stop_condition
                    if stop_condition is None
                    else stop_condition
                ),
                context_provider=(
                    existing.context_provider
                    if context_provider is None
                    else context_provider
                ),
                model_profile=(
                    existing.model_profile
                    if model_profile is None
                    else model_profile
                ),
                verifier=existing.verifier if verifier is None else verifier,
                created_at=existing.created_at,
                updated_at=now,
                metadata=existing.metadata if metadata is None else metadata,
            )
            self._conn.execute("DELETE FROM loop_recipes WHERE recipe_id = ?", (recipe_id,))
            self._insert_recipe(recipe)
            self._conn.commit()
            return recipe

    def delete_recipe(self, recipe_id: str) -> bool:
        if recipe_id == DEFAULT_LOOP_RECIPE_ID:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM loop_recipes WHERE recipe_id = ?",
                (recipe_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def rename_thread(self, thread_id: str, title: str) -> ThreadRecord:
        now = utc_now()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE threads
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (safe_title(title), now, thread_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(thread_id)
            self._conn.commit()
            record = self.get_thread(thread_id)
            if record is None:
                raise KeyError(thread_id)
            return record

    def append_message(
        self,
        thread_id: str,
        *,
        role: str,
        content: str,
        thinking: Optional[dict] = None,
        loop_payload: Optional[dict] = None,
    ) -> ThreadMessage:
        if role not in {"user", "assistant"}:
            raise ValueError("Thread message role must be 'user' or 'assistant'.")
        now = utc_now()
        with self._lock:
            self.ensure_thread(thread_id)
            cursor = self._conn.execute(
                """
                INSERT INTO messages (
                    thread_id, role, content, thinking_json,
                    loop_payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    role,
                    str(content or ""),
                    json_dumps(thinking),
                    json_dumps(loop_payload),
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (now, thread_id),
            )
            self._conn.commit()
            message_id = int(cursor.lastrowid)
            row = self._conn.execute(
                """
                SELECT id, thread_id, role, content, thinking_json,
                       loop_payload_json, created_at
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
            return self._message_from_row(row)

    def append_turn(
        self,
        thread_id: str,
        *,
        user_content: str,
        assistant_content: str,
        thinking: Optional[dict] = None,
        loop_payload: Optional[dict] = None,
        raw_loop_report: Optional[dict] = None,
        public_loop_report: Optional[dict] = None,
        expected_generation: Optional[int] = None,
        expected_instance_id: Optional[str] = None,
        title_if_empty: Optional[str] = None,
    ) -> Optional[tuple[ThreadMessage, ThreadMessage]]:
        if (expected_generation is None) != (expected_instance_id is None):
            raise ValueError(
                "expected_generation and expected_instance_id must be supplied together."
            )
        now = utc_now()
        user_text = str(user_content or "")
        assistant_text = str(assistant_content or "")
        if raw_loop_report is not None and public_loop_report is None:
            raise ValueError(
                "public_loop_report is required when raw_loop_report is supplied."
            )
        thinking_json = json_dumps(thinking)
        loop_payload_json = json_dumps(loop_payload)
        raw_loop_report_json = json_dumps(raw_loop_report)
        public_loop_report_json = json_dumps(public_loop_report)
        loop_run_metadata = (
            loop_run_metadata_from_report(raw_loop_report)
            if raw_loop_report
            else None
        )
        with self._lock:
            thread_row = self._conn.execute(
                """
                SELECT
                    title,
                    instance_id,
                    generation,
                    (
                        SELECT COUNT(*)
                        FROM messages
                        WHERE thread_id = threads.id
                    ) AS message_count
                FROM threads
                WHERE id = ?
                """,
                (thread_id,),
            ).fetchone()
            if thread_row is None:
                if expected_generation is not None or expected_instance_id is not None:
                    return None
                record = self.ensure_thread(thread_id)
                thread_row = {
                    "title": DEFAULT_THREAD_TITLE,
                    "instance_id": record.instance_id,
                    "generation": 0,
                    "message_count": 0,
                }
            if (
                expected_instance_id is not None
                and str(thread_row["instance_id"]) != expected_instance_id
            ):
                return None
            if (
                expected_generation is not None
                and int(thread_row["generation"]) != int(expected_generation)
            ):
                return None
            try:
                if (
                    title_if_empty
                    and int(thread_row["message_count"] or 0) == 0
                    and str(thread_row["title"]) == DEFAULT_THREAD_TITLE
                ):
                    self._conn.execute(
                        """
                        UPDATE threads
                        SET title = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (safe_title(title_if_empty), now, thread_id),
                    )
                user_cursor = self._conn.execute(
                    """
                    INSERT INTO messages (
                        thread_id, role, content, thinking_json,
                        loop_payload_json, created_at
                    )
                    VALUES (?, 'user', ?, NULL, NULL, ?)
                    """,
                    (thread_id, user_text, now),
                )
                assistant_cursor = self._conn.execute(
                    """
                    INSERT INTO messages (
                        thread_id, role, content, thinking_json,
                        loop_payload_json, created_at
                    )
                    VALUES (?, 'assistant', ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        assistant_text,
                        thinking_json,
                        loop_payload_json,
                        now,
                    ),
                )
                if loop_run_metadata and raw_loop_report_json and public_loop_report_json:
                    self._conn.execute(
                        """
                        INSERT INTO loop_runs (
                            run_id, thread_id, user_message_id,
                            assistant_message_id, schema_version,
                            raw_report_json, public_report_json, final_decision,
                            context_provider, backend, model_label, recipe_id,
                            recipe_name, step_count, started_at, completed_at,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            loop_run_metadata["run_id"],
                            thread_id,
                            int(user_cursor.lastrowid),
                            int(assistant_cursor.lastrowid),
                            loop_run_metadata["schema_version"],
                            raw_loop_report_json,
                            public_loop_report_json,
                            loop_run_metadata["final_decision"],
                            loop_run_metadata["context_provider"],
                            loop_run_metadata["backend"],
                            loop_run_metadata["model_label"],
                            loop_run_metadata["recipe_id"],
                            loop_run_metadata["recipe_name"],
                            int(loop_run_metadata["step_count"]),
                            loop_run_metadata["started_at"],
                            loop_run_metadata["completed_at"],
                            now,
                        ),
                    )
                self._conn.execute(
                    "UPDATE threads SET updated_at = ? WHERE id = ?",
                    (now, thread_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            rows = self._conn.execute(
                """
                SELECT id, thread_id, role, content, thinking_json,
                       loop_payload_json, created_at
                FROM messages
                WHERE id IN (?, ?)
                ORDER BY id ASC
                """,
                (int(user_cursor.lastrowid), int(assistant_cursor.lastrowid)),
            ).fetchall()
            if len(rows) != 2:
                raise RuntimeError("Thread turn was not persisted.")
            return tuple(self._message_from_row(row) for row in rows)

    def clear_thread(self, thread_id: str) -> Optional[ThreadRecord]:
        now = utc_now()
        with self._lock:
            if (
                self._conn.execute(
                    "SELECT id FROM threads WHERE id = ?",
                    (thread_id,),
                ).fetchone()
                is None
            ):
                return None
            self._conn.execute("DELETE FROM loop_runs WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            self._conn.execute(
                """
                UPDATE threads
                SET updated_at = ?, generation = generation + 1
                WHERE id = ?
                """,
                (now, thread_id),
            )
            self._conn.commit()
            return self.get_thread(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM threads WHERE id = ?",
                (thread_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def _thread_from_row(
        self,
        row: sqlite3.Row,
        *,
        messages: tuple[ThreadMessage, ...],
        loop_runs: tuple[LoopRunRecord, ...] = (),
    ) -> ThreadRecord:
        return ThreadRecord(
            id=str(row["id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            instance_id=str(row["instance_id"]),
            generation=int(row["generation"] or 0),
            message_count=int(row["message_count"] or len(messages)),
            loop_run_count=int(row["loop_run_count"] or len(loop_runs)),
            messages=messages,
            loop_runs=loop_runs,
            latest=json_loads(row["latest_json"]),
        )

    def _message_from_row(self, row: sqlite3.Row) -> ThreadMessage:
        return ThreadMessage(
            id=int(row["id"]),
            thread_id=str(row["thread_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=str(row["created_at"]),
            thinking=json_loads(row["thinking_json"]),
            loop_payload=json_loads(row["loop_payload_json"]),
        )

    def _loop_run_from_row(self, row: sqlite3.Row) -> LoopRunRecord:
        return LoopRunRecord(
            run_id=str(row["run_id"]),
            thread_id=str(row["thread_id"]),
            user_message_id=(
                int(row["user_message_id"])
                if row["user_message_id"] is not None
                else None
            ),
            assistant_message_id=(
                int(row["assistant_message_id"])
                if row["assistant_message_id"] is not None
                else None
            ),
            schema_version=str(row["schema_version"]),
            raw_report=required_json_loads(row["raw_report_json"]),
            public_report=required_json_loads(row["public_report_json"]),
            final_decision=(
                str(row["final_decision"])
                if row["final_decision"] is not None
                else None
            ),
            context_provider=(
                str(row["context_provider"])
                if row["context_provider"] is not None
                else None
            ),
            backend=str(row["backend"]) if row["backend"] is not None else None,
            model_label=(
                str(row["model_label"]) if row["model_label"] is not None else None
            ),
            recipe_id=str(row["recipe_id"]) if row["recipe_id"] is not None else None,
            recipe_name=(
                str(row["recipe_name"]) if row["recipe_name"] is not None else None
            ),
            step_count=int(row["step_count"] or 0),
            started_at=str(row["started_at"]) if row["started_at"] else None,
            completed_at=str(row["completed_at"]) if row["completed_at"] else None,
            created_at=str(row["created_at"]),
        )

    def _recipe_from_row(self, row: sqlite3.Row) -> LoopRecipe:
        return LoopRecipe(
            recipe_id=str(row["recipe_id"]),
            schema_version=str(row["schema_version"] or LOOP_RECIPE_SCHEMA_VERSION),
            name=str(row["name"]),
            description=str(row["description"] or ""),
            goal=str(row["goal"]),
            instructions=str(row["instructions"] or ""),
            success_criteria=json_list_loads(row["success_criteria_json"]),
            stop_condition=str(row["stop_condition"] or ""),
            context_provider=str(row["context_provider"] or "auto"),
            model_profile=str(row["model_profile"] or "quality"),
            verifier=str(row["verifier"] or "default"),
            metadata=json_loads(row["metadata_json"]) or {},
            created_at=datetime.fromisoformat(
                str(row["created_at"]).replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                str(row["updated_at"]).replace("Z", "+00:00")
            ),
        )

    def _insert_recipe(self, recipe: LoopRecipe) -> None:
        self._conn.execute(
            """
            INSERT INTO loop_recipes (
                recipe_id, schema_version, name, description, goal,
                instructions, success_criteria_json, stop_condition,
                context_provider, model_profile, verifier, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recipe.recipe_id,
                recipe.schema_version,
                recipe.name,
                recipe.description,
                recipe.goal,
                recipe.instructions,
                json_list_dumps(recipe.success_criteria),
                recipe.stop_condition,
                recipe.context_provider,
                recipe.model_profile,
                recipe.verifier,
                json_dumps(dict(recipe.metadata)),
                recipe.created_at.astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                recipe.updated_at.astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
            ),
        )


def validate_vector(vector: list[float] | tuple[float, ...]) -> list[float]:
    values = [float(value) for value in vector]
    if not values:
        raise ValueError("Embedding vector must not be empty.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Embedding vector must contain only finite values.")
    return values


def vector_to_json(vector: list[float] | tuple[float, ...]) -> str:
    return json.dumps(validate_vector(vector), separators=(",", ":"))


def vector_from_json(value: str) -> list[float]:
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("Stored embedding vector must be a list.")
    return validate_vector(data)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
