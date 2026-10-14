"""SQLite 持久化：会话、消息、反馈、查询日志。

为什么需要它
------------
原实现把会话登记表放在进程内存里（`SESSIONS: dict`），后果有三个：
  1. 服务重启，历史全丢；
  2. 多 worker 部署时各进程看到的会话不一致；
  3. 前端只能把聊天记录存在浏览器 localStorage，换设备就没了。

换成 SQLite 的理由：单文件、零运维、自带事务，且足以支撑单机演示与中小规模部署。
若将来要上多实例，只需把这里的连接换成 PostgreSQL —— 表结构与 SQL 基本通用。

设计要点
--------
- **WAL 模式**：允许「读不阻塞写」，避免流式回答期间的读请求被写锁卡住。
- **每请求一个连接**：SQLite 连接不能跨线程共享。用 contextmanager 统一开关，
  既避免线程问题，也不需要连接池。
- **写入用事务**：一次提问要同时写消息、更新会话、写日志，必须原子完成。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '新会话',
    created_at  TEXT NOT NULL,
    last_active TEXT NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0,
    archived    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    meta        TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    message_id  INTEGER,
    rating      TEXT NOT NULL,
    comment     TEXT,
    question    TEXT,
    answer      TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    question      TEXT NOT NULL,
    rewritten     TEXT,
    retrieved     INTEGER NOT NULL DEFAULT 0,
    abstained     INTEGER NOT NULL DEFAULT 0,
    citations     INTEGER NOT NULL DEFAULT 0,
    elapsed_ms    INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_query_log_time ON query_log(created_at);
"""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class SessionRecord:
    id: str
    title: str
    created_at: str
    last_active: str
    turns: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turns": self.turns,
        }


class Database:
    """SQLite 封装。所有写操作都在事务内完成。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ── 连接管理 ──────────────────────────────────────────
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """每次调用开一个新连接。

        SQLite 的连接对象默认不能跨线程使用；FastAPI 的同步端点跑在线程池里，
        复用同一个连接会抛 `SQLite objects created in a thread can only be used
        in that same thread`。每请求一连接是最简单且正确的做法。
        """
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            # WAL 让读写并发不互相阻塞
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def schema_version(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ).fetchone()
            return int(row["value"]) if row else 0

    # ── 会话 ──────────────────────────────────────────────
    def create_session(self, session_id: str, title: str = "新会话") -> SessionRecord:
        stamp = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(id, title, created_at, last_active, turns) "
                "VALUES (?, ?, ?, ?, 0)",
                (session_id, title, stamp, stamp),
            )
        return self.get_session(session_id)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND archived = 0", (session_id,)
            ).fetchone()
        return self._to_session(row) if row else None

    def list_sessions(self, limit: int = 200) -> list[SessionRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE archived = 0 "
                "ORDER BY last_active DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_session(r) for r in rows]

    def touch_session(self, session_id: str, first_message: str = "") -> SessionRecord:
        """登记或更新会话；首次出现时用提问前 24 字作为标题，并把轮数计为 1。"""
        stamp = now_iso()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                title = (first_message or "").strip()[:24] or "新会话"
                conn.execute(
                    "INSERT INTO sessions(id, title, created_at, last_active, turns) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, title, stamp, stamp, 1 if first_message else 0),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET last_active = ?, turns = turns + 1 WHERE id = ?",
                    (stamp, session_id),
                )
        return self.get_session(session_id)  # type: ignore[return-value]

    def rename_session(self, session_id: str, title: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title[:80], session_id)
            )
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def prune_sessions(self, keep: int) -> int:
        """只保留最近活跃的 keep 个会话，返回删除数量。

        防止长时间运行的实例把磁盘写满。
        """
        if keep <= 0:
            return 0
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id NOT IN ("
                "  SELECT id FROM sessions ORDER BY last_active DESC LIMIT ?"
                ")",
                (keep,),
            )
            return cursor.rowcount

    @staticmethod
    def _to_session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            last_active=row["last_active"],
            turns=row["turns"],
        )

    # ── 消息 ──────────────────────────────────────────────
    def add_message(
        self, session_id: str, role: str, content: str, meta: dict | None = None
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    content,
                    now_iso(),
                    json.dumps(meta, ensure_ascii=False) if meta else None,
                ),
            )
            return int(cursor.lastrowid or 0)

    def list_messages(self, session_id: str, limit: int = 500) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
                "meta": json.loads(r["meta"]) if r["meta"] else None,
            }
            for r in rows
        ]

    # ── 反馈 ──────────────────────────────────────────────
    def add_feedback(
        self,
        session_id: str,
        rating: str,
        message_id: int | None = None,
        comment: str = "",
        question: str = "",
        answer: str = "",
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO feedback(session_id, message_id, rating, comment, question, "
                "answer, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, message_id, rating, comment, question, answer, now_iso()),
            )
            return int(cursor.lastrowid or 0)

    def feedback_summary(self) -> dict:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating"
            ).fetchall()
            recent = conn.execute(
                "SELECT rating, question, comment, created_at FROM feedback "
                "ORDER BY id DESC LIMIT 20"
            ).fetchall()
        counts = {r["rating"]: r["n"] for r in rows}
        total = sum(counts.values())
        positive = counts.get("up", 0)
        return {
            "total": total,
            "up": positive,
            "down": counts.get("down", 0),
            "positive_rate": round(positive / total, 4) if total else None,
            "recent": [dict(r) for r in recent],
        }

    # ── 查询日志 ──────────────────────────────────────────
    def log_query(
        self,
        session_id: str,
        question: str,
        rewritten: str = "",
        retrieved: int = 0,
        abstained: bool = False,
        citations: int = 0,
        elapsed_ms: int = 0,
        error: str = "",
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO query_log(session_id, question, rewritten, retrieved, "
                "abstained, citations, elapsed_ms, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    question[:2000],
                    (rewritten or "")[:2000],
                    retrieved,
                    1 if abstained else 0,
                    citations,
                    elapsed_ms,
                    (error or "")[:500],
                    now_iso(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def metrics(self) -> dict:
        """运行指标：总量、拒答率、平均耗时、引用数。"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(abstained) AS abstained, "
                "AVG(elapsed_ms) AS avg_ms, "
                "AVG(retrieved) AS avg_retrieved, "
                "AVG(citations) AS avg_citations, "
                "SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END) AS errors "
                "FROM query_log"
            ).fetchone()
            sessions = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
            docs = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]

        total = row["total"] or 0
        abstained = row["abstained"] or 0
        return {
            "total_queries": total,
            "abstained": abstained,
            "abstain_rate": round(abstained / total, 4) if total else None,
            "errors": row["errors"] or 0,
            "avg_elapsed_ms": round(row["avg_ms"], 1) if row["avg_ms"] else None,
            "avg_retrieved": round(row["avg_retrieved"], 2) if row["avg_retrieved"] else None,
            "avg_citations": round(row["avg_citations"], 2) if row["avg_citations"] else None,
            "sessions": sessions,
            "messages": docs,
        }

    def recent_queries(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT question, rewritten, retrieved, abstained, citations, "
                "elapsed_ms, error, created_at FROM query_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


_db_singleton: Database | None = None
_db_lock = threading.Lock()


def get_database(path: Path | None = None) -> Database:
    """取数据库单例。未显式给路径时读全局配置。"""
    global _db_singleton
    if _db_singleton is not None and path is None:
        return _db_singleton
    with _db_lock:
        if _db_singleton is None or path is not None:
            if path is None:
                try:
                    from .config import settings
                except ImportError:
                    from config import settings

                path = settings.database_path
            _db_singleton = Database(path)
    return _db_singleton


def reset_database() -> None:
    """释放单例（测试用）。"""
    global _db_singleton
    _db_singleton = None


__all__ = [
    "Database",
    "SessionRecord",
    "get_database",
    "reset_database",
    "now_iso",
    "Any",
]
