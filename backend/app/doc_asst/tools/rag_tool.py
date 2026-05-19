from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Set, Type

from openai import OpenAI
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, PrivateAttr

from ..config import ENV
from ..persistence.pool import get_pool
from .base_tool import DocasstBaseTool


class RAGArgs(BaseModel):
    query: str = Field(description="知识库检索查询文本")
    top_k: int = Field(default=5, description="返回结果数量上限")


class RAGTool(DocasstBaseTool):
    """
    PostgreSQL + pgvector RAG 检索工具（只查库）：
    - 查询向量化
    - 候选召回
    - 重排序输出 top_k
    """

    name: str = "rag_search"
    description: str = "基于 pgvector 的本地知识库检索工具。用于查询项目内部文档、已有资料、历史记录。"
    args_schema: Type[BaseModel] = RAGArgs

    postgres_dsn: str = Field(default="", exclude=True)
    embedding_model: str = "text-embedding-3-small"
    default_top_k: int = 5
    _client: Any = PrivateAttr(default=None)
    _pool: Any = PrivateAttr(default=None)

    def __init__(self, postgres_dsn: str | None = None, embedding_model: str | None = None, default_top_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        if postgres_dsn:
            self.postgres_dsn = postgres_dsn
        elif os.getenv("LANGGRAPH_POSTGRES_DSN"):
            self.postgres_dsn = os.getenv("LANGGRAPH_POSTGRES_DSN", "")
        if embedding_model:
            self.embedding_model = embedding_model
        elif os.getenv("EMBEDDING_MODEL_ID"):
            self.embedding_model = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
        self.default_top_k = default_top_k
        self._client = OpenAI(api_key=ENV.llm_api_key, base_url=ENV.llm_base_url, timeout=ENV.llm_timeout)
        # 使用全局连接池，不再维护裸连
        self._pool = get_pool()

    def _get_conn(self):
        """从连接池获取连接（上下文管理器，自动归还）。"""
        if self._pool is None:
            raise ValueError("缺少 LANGGRAPH_POSTGRES_DSN，无法使用 pgvector 检索")
        return self._pool.connection()

    def _execute(self, query: str = "", top_k: int = 5, **kwargs) -> Dict[str, Any]:
        query = str(query).strip()
        top_k = int(top_k) or self.default_top_k
        if not query:
            raise ValueError("rag_search 工具需要提供 query 参数")
        if self._pool is None:
            raise ValueError("缺少 LANGGRAPH_POSTGRES_DSN，无法使用 pgvector 检索")

        normalized_query = self._normalize_query(query)
        query_terms = self._tokenize_query(normalized_query)
        query_vector = self._embed_text(normalized_query)
        candidate_k = top_k * 4

        with self._pool.connection() as conn:
            candidates = self._retrieve_candidates(conn, query_vector=query_vector, candidate_k=candidate_k)

        rows = self._rerank(candidates=candidates, query_terms=query_terms, top_k=top_k)

        results = [
            {"title": row["title"], "source_path": row["source_path"], "snippet": row["content"], "score": row["final_score"], "semantic_score": row["semantic_score"], "keyword_score": row["keyword_score"], "source": "pgvector"}
            for row in rows
        ]
        return {"query": query, "normalized_query": normalized_query, "top_k": top_k, "vector_backend": "postgres_pgvector", "results": results, "candidate_count": len(candidates)}

    def compact(self, result: Any) -> str:
        """精简：最多3条结果，保留 title + source_path + snippet(300字)"""
        if isinstance(result, dict) and "results" in result:
            compact_results = []
            for item in result["results"][:3]:
                compact_results.append({
                    "title": item.get("title", ""),
                    "source_path": item.get("source_path", ""),
                    "snippet": str(item.get("snippet", ""))[:300],
                })
            return json.dumps({"ok": True, "results": compact_results}, ensure_ascii=False)
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False, default=str)

    def _retrieve_candidates(self, conn: psycopg.Connection, query_vector: List[float], candidate_k: int) -> List[Dict[str, Any]]:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, source_path, content, (1 - (embedding <=> %s::vector)) AS semantic_score FROM rag_chunks ORDER BY embedding <=> %s::vector LIMIT %s;",
                (self._vector_literal(query_vector), self._vector_literal(query_vector), candidate_k),
            )
            rows = cur.fetchall()
        return [{"title": str(row[0]), "source_path": str(row[1]), "content": str(row[2]), "semantic_score": float(row[3])} for row in rows]

    def _rerank(self, candidates: List[Dict[str, Any]], query_terms: Set[str], top_k: int) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []
        for row in candidates:
            semantic_score = float(row.get("semantic_score", 0.0))
            keyword_score = self._keyword_overlap_score(query_terms, row.get("content", ""))
            final_score = (semantic_score * 0.8) + (keyword_score * 0.2)
            reranked.append({**row, "keyword_score": keyword_score, "final_score": final_score})
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]

    def _normalize_query(self, query: str) -> str:
        query = query.strip().lower()
        query = re.sub(r"\s+", " ", query)
        return query

    def _tokenize_query(self, query: str) -> Set[str]:
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", query)
        return {t for t in tokens if len(t) >= 2}

    def _keyword_overlap_score(self, query_terms: Set[str], content: str) -> float:
        if not query_terms:
            return 0.0
        normalized_content = content.lower()
        hit = sum(1 for term in query_terms if term in normalized_content)
        return hit / len(query_terms)

    def _embed_text(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self.embedding_model, input=text)
        return list(resp.data[0].embedding)

    def _vector_literal(self, vec: List[float]) -> str:
        return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
