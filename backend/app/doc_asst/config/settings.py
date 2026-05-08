from __future__ import annotations

from typing import Dict, List

# 线程 ID 是 LangGraph 的会话隔离键；相同 thread_id 可复用记忆/检查点。
DEFAULT_THREAD_ID = "default-thread"

# 功能开关：V1 可以按需逐个关闭，便于定位问题。
ENABLE_SEARCH = True
ENABLE_RAG = True
ENABLE_MEMORY = True
ENABLE_POSTGRES_CHECKPOINT = False

# 业务规则：按角色分配工具，避免所有 Agent 拥有全部能力。
AGENT_TOOL_POLICY: Dict[str, List[str]] = {
    "planner_agent": ((["web_search"] if ENABLE_SEARCH else []) + (["memory_store"] if ENABLE_MEMORY else [])),
    "extractor_summarizer_agent": (
        (["web_search"] if ENABLE_SEARCH else []) + (["rag_search"] if ENABLE_RAG else [])
    ),
    "reporter_agent": (["memory_store"] if ENABLE_MEMORY else []),
}
