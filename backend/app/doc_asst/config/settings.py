from __future__ import annotations

from typing import Dict, List

# 线程 ID 是 LangGraph 的会话隔离键；相同 thread_id 可复用记忆/检查点。
DEFAULT_THREAD_ID = "default-thread"

# 功能开关：V1 可以按需逐个关闭，便于定位问题。
ENABLE_SEARCH = True
ENABLE_RAG = True
ENABLE_MEMORY = True
ENABLE_POSTGRES_CHECKPOINT = True
ENABLE_FILE_CONTENT_READER = True

# 业务规则：按角色分配工具，避免所有 Agent 拥有全部能力。
AGENT_TOOL_POLICY: Dict[str, List[str]] = {
    "planner_agent": (["web_search"] if ENABLE_SEARCH else []),
    "extractor_summarizer_agent": (
        (["web_search"] if ENABLE_SEARCH else []) + (["rag_search"] if ENABLE_RAG else []) + (["file_content_reader"] if ENABLE_FILE_CONTENT_READER else [])
    ),
    "reporter_agent": (["memory_store"] if ENABLE_MEMORY else []),
}

# ReAct 循环滑动窗口：保留最近 N 轮工具交互（assistant+tool），0 表示不限制。
REACT_SLIDING_WINDOW = 3

# 单个 Agent 内 ReAct 工具调用最大轮次：超过则强制生成最终回答。
MAX_TOOL_ROUNDS = 4

# 反思迭代最大轮次：整个 planner→extractor→reporter→reflection 循环最多执行几轮。
MAX_REFLECTION_ITERATIONS = 2
