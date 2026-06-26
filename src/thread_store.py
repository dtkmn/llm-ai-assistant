from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_THREAD_TITLE = "New thread"
MAX_THREAD_TITLE_LENGTH = 64


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


def safe_title(value: str | None) -> str:
    title = " ".join(str(value or "").split()).strip()
    if not title:
        return DEFAULT_THREAD_TITLE
    return title[:MAX_THREAD_TITLE_LENGTH]


def json_dumps(value: dict | None) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_loads(value: str | None) -> Optional[dict]:
    if not value:
        return None
    data = json.loads(value)
    return data if isinstance(data, dict) else None


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
    messages: tuple[ThreadMessage, ...] = ()
    latest: Optional[dict] = None

    def summary_dict(self, *, include_latest: bool = False) -> dict:
        payload = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }
        if include_latest:
            payload["latest"] = self.latest
        return payload

    def detail_dict(self) -> dict:
        payload = self.summary_dict(include_latest=True)
        payload["messages"] = [message.to_dict() for message in self.messages]
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
            return self._thread_from_row(row, messages=messages)

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
        thinking_json = json_dumps(thinking)
        loop_payload_json = json_dumps(loop_payload)
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
    ) -> ThreadRecord:
        return ThreadRecord(
            id=str(row["id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            instance_id=str(row["instance_id"]),
            generation=int(row["generation"] or 0),
            message_count=int(row["message_count"] or len(messages)),
            messages=messages,
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
