from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
from typing import Any, Dict, List

import psycopg
from openai import OpenAI

from ..config import ENV
from ..persistence.pool import get_pool


class RAGIngestor:
    """
    上传文件到 PostgreSQL 的独立入库器（不依赖 rag_tool）：
    1) 原文写入 rag_documents
    2) 文本切分
    3) 通过嵌入器生成向量
    4) 向量写入 rag_chunks(pgvector)

    注意：
    - 不自动建表，默认目标表已存在。
    """

    def __init__(
        self,
        postgres_dsn: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        chunk_size: int = 800,
    ) -> None:
        self.postgres_dsn = postgres_dsn or os.getenv("LANGGRAPH_POSTGRES_DSN", "")
        if not self.postgres_dsn:
            raise ValueError("缺少 LANGGRAPH_POSTGRES_DSN")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
        self.embedding_dim = int(embedding_dim or os.getenv("EMBEDDING_DIM", "1536"))
        self.chunk_size = chunk_size
        self.client = OpenAI(
            api_key=ENV.llm_api_key,
            base_url=ENV.llm_base_url,
            timeout=ENV.llm_timeout,
        )
        # 支持纯文本 + 常见办公文档（pdf/docx）。
        self.allowed_suffixes = {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".pdf", ".docx"}

    def ingest_file(
        self,
        file_path: str,
        tenant_id: str = "default",
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if path.suffix.lower() not in self.allowed_suffixes:
            raise ValueError(f"当前支持的文件类型: {sorted(self.allowed_suffixes)}")

        content = self._read_content(path)
        title = path.stem
        source_path = str(path)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        doc_id = hashlib.sha256(f"{tenant_id}:{source_path}".encode("utf-8")).hexdigest()
        metadata_obj = metadata or {}
        chunks = self._split_text(content)

        pool = get_pool()
        if pool is not None:
            with pool.connection() as conn:
                self._upsert_document(
                    conn=conn,
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    source_path=source_path,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    metadata=metadata_obj,
                )
                self._upsert_chunks(
                    conn=conn,
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    source_path=source_path,
                    title=title,
                    chunks=chunks,
                    metadata=metadata_obj,
                )
        else:
            # 回退：无连接池时使用裸连
            with psycopg.connect(self.postgres_dsn) as conn:
                self._upsert_document(
                    conn=conn,
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    source_path=source_path,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    metadata=metadata_obj,
                )
                self._upsert_chunks(
                    conn=conn,
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    source_path=source_path,
                    title=title,
                    chunks=chunks,
                    metadata=metadata_obj,
                )

        return {
            "ok": True,
            "doc_id": doc_id,
            "tenant_id": tenant_id,
            "source_path": source_path,
            "stored_document": True,
            "stored_chunks": len(chunks),
        }

    def _upsert_document(
        self,
        conn: psycopg.Connection,
        doc_id: str,
        tenant_id: str,
        source_path: str,
        title: str,
        content: str,
        content_hash: str,
        metadata: Dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_documents
                    (doc_id, tenant_id, source_path, title, content, content_hash, version, metadata, created_at, updated_at)
                VALUES
                    (%s, %s, %s, %s, %s, %s, 1, %s::jsonb, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                SET title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    metadata = EXCLUDED.metadata,
                    version = rag_documents.version + 1,
                    updated_at = EXCLUDED.updated_at;
                """,
                (doc_id, tenant_id, source_path, title, content, content_hash, json.dumps(metadata, ensure_ascii=False), now, now),
            )
        conn.commit()

    def _upsert_chunks(
        self,
        conn: psycopg.Connection,
        doc_id: str,
        tenant_id: str,
        source_path: str,
        title: str,
        chunks: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        """
        向量存储职责：把 chunk + embedding 写入 rag_chunks(pgvector)。
        """
        if not chunks:
            return
        vectors = self._embed_texts(chunks)
        now = datetime.now(timezone.utc)
        with conn.cursor() as cur:
            for idx, text in enumerate(chunks):
                chunk_key = f"{doc_id}:{idx}:{text}"
                chunk_id = hashlib.sha256(chunk_key.encode("utf-8")).hexdigest()
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    INSERT INTO rag_chunks
                        (chunk_id, source_path, title, chunk_index, content, content_hash, embedding, updated_at)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding,
                        title = EXCLUDED.title,
                        updated_at = EXCLUDED.updated_at;
                    """,
                    (
                        chunk_id,
                        source_path,
                        title,
                        idx,
                        text,
                        content_hash,
                        self._vector_literal(vectors[idx]),
                        now,
                    ),
                )
        conn.commit()

    def _split_text(self, text: str) -> List[str]:
        """
        切分职责：先按段落分，再按固定窗口切片，确保可控粒度。
        """
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        parts: List[str] = []
        for block in blocks:
            if len(block) <= self.chunk_size:
                parts.append(block)
                continue
            for i in range(0, len(block), self.chunk_size):
                piece = block[i : i + self.chunk_size].strip()
                if piece:
                    parts.append(piece)
        return parts

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入器职责：仅负责把文本列表转成向量列表。
        """
        resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [list(item.embedding) for item in resp.data]

    def _vector_literal(self, vec: List[float]) -> str:
        return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

    def _read_content(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown", ".txt", ".rst", ".csv", ".json"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".docx":
            return self._read_docx(path)
        raise ValueError(f"暂不支持的文件类型: {suffix}")

    def _read_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc

        reader = PdfReader(str(path))
        texts: List[str] = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n\n".join(t.strip() for t in texts if t and t.strip())

    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document
        except Exception as exc:
            raise RuntimeError("解析 DOCX 需要安装 python-docx：pip install python-docx") from exc

        doc = Document(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(parts)
