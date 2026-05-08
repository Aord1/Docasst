from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List

import psycopg

from .base_tool import BaseTool


class MemoryTool(BaseTool):
    """
    轻量记忆工具（PostgreSQL）：
    - save: 保存键值记忆
    - recall: 按 scope/thread_id + namespace 读取最近记忆
    """

    name = "memory_store"
    description = "持久化记忆工具，支持 save/recall。"

    def __init__(self, postgres_dsn: str | None = None) -> None:
        self.postgres_dsn = postgres_dsn or os.getenv("LANGGRAPH_POSTGRES_DSN", "")

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        action = str(kwargs.get("action", "")).strip().lower()
        scope = str(kwargs.get("scope", "default-thread"))
        namespace = str(kwargs.get("namespace", "writer"))
        key = str(kwargs.get("key", "")).strip()
        limit = int(kwargs.get("limit", 5))

        if not self.postgres_dsn:
            raise ValueError("缺少 LANGGRAPH_POSTGRES_DSN，无法使用 memory_store")

        with psycopg.connect(self.postgres_dsn) as conn:
            self._ensure_schema(conn)
            if action == "save":
                value = kwargs.get("value")
                if not key:
                    raise ValueError("save 动作需要 key")
                self._save(conn, scope=scope, namespace=namespace, key=key, value=value)
                return {"action": "save", "scope": scope, "namespace": namespace, "key": key, "saved": True}

            if action == "recall":
                rows = self._recall(conn, scope=scope, namespace=namespace, key=key or None, limit=limit)
                return {"action": "recall", "scope": scope, "namespace": namespace, "items": rows}

            raise ValueError("memory_store action 仅支持 save 或 recall")

    def _ensure_schema(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            # scope + namespace + key 组合定义一类记忆数据。
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    id BIGSERIAL PRIMARY KEY,
                    scope TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_scope_ns_key
                ON memory_records (scope, namespace, key, updated_at DESC);
                """
            )
        conn.commit()

    def _save(
        self,
        conn: psycopg.Connection,
        scope: str,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_records (scope, namespace, key, value_json, updated_at)
                VALUES (%s, %s, %s, %s::jsonb, %s);
                """,
                (
                    scope,
                    namespace,
                    key,
                    json.dumps(value, ensure_ascii=False),
                    datetime.now(timezone.utc),
                ),
            )
        conn.commit()

    def _recall(
        self,
        conn: psycopg.Connection,
        scope: str,
        namespace: str,
        key: str | None,
        limit: int,
    ) -> List[Dict[str, Any]]:
        with conn.cursor() as cur:
            # key 为空时按 namespace 回看最近记忆；有 key 时读取同键历史版本。
            if key:
                cur.execute(
                    """
                    SELECT key, value_json, updated_at
                    FROM memory_records
                    WHERE scope = %s AND namespace = %s AND key = %s
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (scope, namespace, key, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT key, value_json, updated_at
                    FROM memory_records
                    WHERE scope = %s AND namespace = %s
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (scope, namespace, limit),
                )
            rows = cur.fetchall()

        return [
            {"key": str(row[0]), "value": row[1], "updated_at": row[2].isoformat() if row[2] else None}
            for row in rows
        ]
