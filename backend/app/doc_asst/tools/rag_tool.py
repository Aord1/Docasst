from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os
import re
from typing import Any, Dict, List

from openai import OpenAI
import psycopg

from ..config import ENV
from .base_tool import BaseTool


@dataclass
class _Chunk:
    chunk_id: str
    source_path: str
    title: str
    chunk_index: int
    text: str
    content_hash: str


class RAGTool(BaseTool):
    """
    PostgreSQL + pgvector RAG 工具：
    - 读取本地文档
    - 生成 embedding 并写入 pgvector
    - 用向量相似度召回 top_k 片段
    """

    name = "rag_search"
    description = "基于 pgvector 的本地知识库检索工具。"

    def __init__(
        self,
        source_dir: str | None = None,
        postgres_dsn: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        default_top_k: int = 5,
    ) -> None:
        self.source_dir = Path(source_dir or os.getenv("RAG_SOURCE_DIR", "workspace/sources"))
        self.postgres_dsn = postgres_dsn or os.getenv("LANGGRAPH_POSTGRES_DSN", "")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
        self.embedding_dim = int(embedding_dim or os.getenv("EMBEDDING_DIM", "1536"))
        self.default_top_k = default_top_k
        self.client = OpenAI(
            api_key=ENV.llm_api_key,
            base_url=ENV.llm_base_url,
            timeout=ENV.llm_timeout,
        )

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        top_k = int(kwargs.get("top_k", self.default_top_k))
        if not query:
            raise ValueError("rag_search 工具需要提供 query 参数")
        if not self.postgres_dsn:
            raise ValueError("缺少 LANGGRAPH_POSTGRES_DSN，无法使用 pgvector 检索")

        # 每次查询前同步本地资料到向量表：V1 用“边查边增量索引”，实现简单可靠。
        chunks = self._load_chunks()
        query_vector = self._embed_text(query)

        with psycopg.connect(self.postgres_dsn) as conn:
            self._ensure_schema(conn)
            self._upsert_chunks(conn, chunks)
            rows = self._search(conn, query_vector=query_vector, top_k=top_k)

        results = [
            {
                "title": row["title"],
                "source_path": row["source_path"],
                "snippet": row["content"],
                "score": row["score"],
                "source": "pgvector",
            }
            for row in rows
        ]
        return {
            "query": query,
            "top_k": top_k,
            "source_dir": str(self.source_dir),
            "vector_backend": "postgres_pgvector",
            "results": results,
            "indexed_chunks": len(chunks),
        }

    def _ensure_schema(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            # vector 扩展和表结构在首次调用自动初始化。
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding VECTOR({self.embedding_dim}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
        conn.commit()

    def _upsert_chunks(self, conn: psycopg.Connection, chunks: List[_Chunk]) -> None:
        if not chunks:
            return

        existing: Dict[str, str] = {}
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, content_hash FROM rag_chunks;")
            for row in cur.fetchall():
                existing[str(row[0])] = str(row[1])

        with conn.cursor() as cur:
            for chunk in chunks:
                # hash 未变化则跳过 embedding，减少无效开销。
                if existing.get(chunk.chunk_id) == chunk.content_hash:
                    continue
                vector = self._embed_text(chunk.text)
                cur.execute(
                    """
                    INSERT INTO rag_chunks
                        (chunk_id, source_path, title, chunk_index, content, content_hash, embedding, updated_at)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        title = EXCLUDED.title,
                        chunk_index = EXCLUDED.chunk_index,
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding,
                        updated_at = EXCLUDED.updated_at;
                    """,
                    (
                        chunk.chunk_id,
                        chunk.source_path,
                        chunk.title,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.content_hash,
                        self._vector_literal(vector),
                        datetime.now(timezone.utc),
                    ),
                )
        conn.commit()

    def _search(self, conn: psycopg.Connection, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    title,
                    source_path,
                    content,
                    (1 - (embedding <=> %s::vector)) AS score
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (self._vector_literal(query_vector), self._vector_literal(query_vector), top_k),
            )
            rows = cur.fetchall()

        return [
            {
                "title": str(row[0]),
                "source_path": str(row[1]),
                "content": str(row[2]),
                "score": float(row[3]),
            }
            for row in rows
        ]

    def _load_chunks(self) -> List[_Chunk]:
        if not self.source_dir.exists():
            return []

        chunks: List[_Chunk] = []
        for file_path in self.source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".md", ".txt"}:
                continue

            text = file_path.read_text(encoding="utf-8", errors="ignore")
            parts = self._split_text(text)
            for index, part in enumerate(parts):
                key = f"{file_path}:{index}:{part}"
                content_hash = hashlib.sha256(part.encode("utf-8")).hexdigest()
                chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
                chunks.append(
                    _Chunk(
                        chunk_id=chunk_id,
                        source_path=str(file_path),
                        title=file_path.stem,
                        chunk_index=index,
                        text=part,
                        content_hash=content_hash,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> List[str]:
        # 先按空行切段，再对长段切窗，平衡语义完整性和召回粒度。
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        parts: List[str] = []
        for block in blocks:
            if len(block) <= 800:
                parts.append(block)
                continue
            for i in range(0, len(block), 800):
                piece = block[i : i + 800].strip()
                if piece:
                    parts.append(piece)
        return parts

    def _embed_text(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=text)
        return list(resp.data[0].embedding)

    def _vector_literal(self, vec: List[float]) -> str:
        return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
