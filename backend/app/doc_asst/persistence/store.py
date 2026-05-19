from __future__ import annotations

import os
from typing import Optional

from langgraph.store.base import BaseStore

from .pool import get_pool


def get_store() -> Optional[BaseStore]:
    """
    获取全局复用的 LangGraph Store 实例。
    首次调用时创建并初始化，后续调用直接复用同一实例。

    - 配置了 LANGGRAPH_POSTGRES_DSN 时使用 PostgresStore（持久化 + 连接池）
    - 否则使用 InMemoryStore（进程内，重启丢失）

    使用共享连接池，避免裸连和连接泄漏。

    Returns:
        BaseStore 实例。
    """
    # 全局单例：首次创建后缓存，后续复用
    if hasattr(get_store, "_instance"):
        return get_store._instance

    pool = get_pool()
    index_cfg = _index_config()

    if pool is not None:
        from langgraph.store.postgres import PostgresStore

        # 直接传入连接池构造，复用全局池
        store: BaseStore = PostgresStore(conn=pool, index=index_cfg)
        store.setup()
    else:
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore(index=index_cfg)

    get_store._instance = store
    return store


def _index_config():
    """构建 Store 的向量索引配置（用于语义搜索）。"""
    from openai import OpenAI
    from ..config import ENV

    embedding_model = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

    client = OpenAI(api_key=ENV.llm_api_key, base_url=ENV.llm_base_url, timeout=ENV.llm_timeout)

    def embed_texts(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=embedding_model, input=texts)
        return [list(item.embedding) for item in resp.data]

    return {"dims": embedding_dim, "embed": embed_texts, "fields": ["text"]}
