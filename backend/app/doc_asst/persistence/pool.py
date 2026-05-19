"""全局 PostgreSQL 连接池单例。"""
from __future__ import annotations

import atexit
import logging
import os
from typing import Optional

from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# 默认池参数
_DEFAULT_MIN_SIZE = 2
_DEFAULT_MAX_SIZE = 10


def get_pool() -> Optional[ConnectionPool]:
    """
    获取全局复用的 PostgreSQL 连接池。

    - 首次调用时创建并打开连接池，后续调用直接复用。
    - 通过 ``atexit`` 注册关闭回调，进程退出时自动关闭池。
    - 未配置 ``LANGGRAPH_POSTGRES_DSN`` 时返回 None。

    环境变量:
        LANGGRAPH_POSTGRES_DSN: PostgreSQL 连接串
        PG_POOL_MIN_SIZE: 最小连接数（默认 2）
        PG_POOL_MAX_SIZE: 最大连接数（默认 10）
    """
    dsn = os.getenv("LANGGRAPH_POSTGRES_DSN")
    if not dsn:
        return None

    if hasattr(get_pool, "_instance"):
        return get_pool._instance

    min_size = int(os.getenv("PG_POOL_MIN_SIZE", str(_DEFAULT_MIN_SIZE)))
    max_size = int(os.getenv("PG_POOL_MAX_SIZE", str(_DEFAULT_MAX_SIZE)))

    pool = ConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
        open=True,
    )

    # 注册退出时关闭连接池
    atexit.register(_close_pool, pool)

    get_pool._instance = pool
    logger.info("PostgreSQL 连接池已创建 (min=%d, max=%d)", min_size, max_size)
    return pool


def _close_pool(pool: ConnectionPool) -> None:
    """进程退出时关闭连接池。"""
    try:
        pool.close()
        logger.info("PostgreSQL 连接池已关闭")
    except Exception:  # noqa: BLE001
        pass
