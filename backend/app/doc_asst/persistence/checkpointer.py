from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator


@contextmanager
def postgres_checkpointer() -> Generator[object, None, None]:
    """
    创建 LangGraph 的 PostgreSQL checkpointer（上下文管理器）。
    需要环境变量:
    - LANGGRAPH_POSTGRES_DSN
      示例: postgresql://user:password@localhost:5432/docasst
    """

    dsn = os.getenv("LANGGRAPH_POSTGRES_DSN")
    if not dsn:
        raise ValueError("缺少 LANGGRAPH_POSTGRES_DSN，无法启用 PostgreSQL 持久化。")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ImportError(
            "未安装相关依赖。"
        ) from exc

    with PostgresSaver.from_conn_string(dsn) as checkpointer:
        # 首次运行可初始化必要表结构；重复调用通常是幂等的。
        checkpointer.setup()
        yield checkpointer
