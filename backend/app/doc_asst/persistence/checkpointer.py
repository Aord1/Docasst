from __future__ import annotations

import os
from typing import Optional

from langgraph.checkpoint.postgres import PostgresSaver

from .pool import get_pool


def get_checkpointer() -> Optional[PostgresSaver]:
    """
    获取全局复用的 PostgreSQL checkpointer 实例。
    首次调用时创建并初始化表结构，后续调用直接复用同一实例。

    使用共享连接池，避免裸连和连接泄漏。

    需要环境变量:
    - LANGGRAPH_POSTGRES_DSN
      示例: postgresql://user:password@localhost:5432/docasst

    Returns:
        PostgresSaver 实例，若未配置 DSN 则返回 None。
    """
    pool = get_pool()
    if pool is None:
        return None

    # 全局单例：首次创建后缓存，后续复用
    if not hasattr(get_checkpointer, "_instance"):
        checkpointer = PostgresSaver(conn=pool)
        checkpointer.setup()
        get_checkpointer._instance = checkpointer

    return get_checkpointer._instance
