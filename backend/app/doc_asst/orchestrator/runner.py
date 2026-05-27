from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from langchain_openai import ChatOpenAI

from ..agents import ExtractorSummarizerAgent, PlannerAgent, ReflectionAgent, ReporterAgent
from ..config.settings import (
    DEFAULT_THREAD_ID,
    ENABLE_FILE_CONTENT_READER,
    ENABLE_MEMORY,
    ENABLE_POSTGRES_CHECKPOINT,
    ENABLE_RAG,
    ENABLE_SEARCH,
    MAX_REFLECTION_ITERATIONS,
)
from ..core.llm import create_chat_model
from ..persistence.checkpointer import get_checkpointer
from ..persistence.store import get_store as get_global_store
from ..tools import FileContentTool, MemoryTool, RAGTool, SearchTool
from .graph import build_workflow_graph
from .state import WorkflowState

# 外层工作流节点名集合，用于从子图消息元数据中反推所属节点
_OUTER_NODES = frozenset({
    "simpleRouter", "planner", "extractor_summarizer", "reporter", "reflection",
})


def _resolve_tools_for_agent(agent_name: str, available_tools: List) -> List:
    """根据 agent 名称从可用工具列表中解析所需工具。"""
    from ..config.settings import AGENT_TOOL_POLICY
    tool_names = AGENT_TOOL_POLICY.get(agent_name, [])
    tool_map = {t.name: t for t in available_tools}
    return [tool_map[name] for name in tool_names if name in tool_map]


def _extract_chunk_text(chunk: Any) -> str:
    """从 AIMessageChunk 提取纯文本内容，跳过 tool_calls 等非文本块。"""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def _derive_outer_node(metadata: dict, active_node: str | None = None) -> str:
    """
    从消息元数据推断外层工作流节点名。

    create_react_agent 子图中的 LLM 调用，其 langgraph_checkpoint_ns
    通常以父节点名开头（如 "planner|..." 或 "planner:..."），
    据此将 token 事件归到正确的外层节点。
    """
    namespace = metadata.get("langgraph_checkpoint_ns", "")
    if namespace:
        for sep in ("|", ":"):
            if sep in namespace:
                candidate = namespace.split(sep)[0]
                if candidate in _OUTER_NODES:
                    return candidate
        if namespace in _OUTER_NODES:
            return namespace
    node = metadata.get("langgraph_node", "")
    if node in _OUTER_NODES:
        return node
    return active_node or node or "unknown"


def _build_runtime(
    user_input: str,
    memory: Optional[List[Dict[str, str]]] = None,
    tools: Optional[List] = None,
    chat_model: Optional[ChatOpenAI] = None,
    thread_id: str = DEFAULT_THREAD_ID,
    verbose: bool = False,
    max_iterations: int = MAX_REFLECTION_ITERATIONS,
    uploaded_files: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], WorkflowState, Dict[str, Any]]:
    # 创建统一的 ChatOpenAI 实例
    model = chat_model or create_chat_model()

    # 构建可用工具列表
    default_tools: List = []
    if ENABLE_SEARCH:
        default_tools.append(SearchTool())
    if ENABLE_RAG:
        default_tools.append(RAGTool())
    if ENABLE_MEMORY:
        default_tools.append(MemoryTool())
    if ENABLE_FILE_CONTENT_READER:
        default_tools.append(FileContentTool())
    available_tools: List = tools or default_tools

    # 按 agent 角色分配工具
    planner_tools = _resolve_tools_for_agent("planner_agent", available_tools)
    extractor_tools = _resolve_tools_for_agent("extractor_summarizer_agent", available_tools)
    reporter_tools = _resolve_tools_for_agent("reporter_agent", available_tools)

    # 创建 agent 实例（全部使用 ChatOpenAI）
    planner = PlannerAgent(llm=model, tools=planner_tools)
    extractor = ExtractorSummarizerAgent(llm=model, tools=extractor_tools)
    reporter = ReporterAgent(llm=model, tools=reporter_tools)
    # Reflection 无工具，也使用 ChatOpenAI
    reflector = ReflectionAgent(llm=model, tools=[])

    initial_messages: List[Dict[str, str]] = list(memory or [])
    initial_messages.append({"role": "user", "content": user_input})

    initial_state: WorkflowState = {
        "messages": initial_messages,
        "user_input": user_input,
        "memory": memory,
        "plan_text": "",
        "summary_text": "",
        "final_text": "",
        "reflection_text": "",
        "reflection_output": {},
        "is_simple_query": False,
        "simple_query_score": 0.0,
        "simple_query_reason": "",
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "iteration": 1,
        "max_iterations": max(1, int(max_iterations)),
    }

    # 运行时元数据通过 LangGraph config 传递
    configurable: Dict[str, Any] = {
        "thread_id": thread_id,
        "verbose": verbose,
    }
    if uploaded_files:
        configurable["uploaded_files"] = uploaded_files
    config = {"configurable": configurable}

    agent_bundle = {
        "planner": planner,
        "extractor": extractor,
        "reporter": reporter,
        "reflector": reflector,
    }
    return agent_bundle, initial_state, config


def run_workflow_stream(
    user_input: str,
    memory: Optional[List[Dict[str, str]]] = None,
    tools: Optional[List] = None,
    chat_model: Optional[ChatOpenAI] = None,
    thread_id: str = DEFAULT_THREAD_ID,
    use_postgres_checkpoint: bool = ENABLE_POSTGRES_CHECKPOINT,
    verbose: bool = False,
    max_iterations: int = MAX_REFLECTION_ITERATIONS,
    stream_mode: Union[str, List[str], None] = "updates",
    uploaded_files: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    LangGraph 流式执行。

    stream_mode="updates" 时仅产出节点完成事件（向后兼容）。
    stream_mode=["messages", "updates"] 时同时产出 LLM token 级事件，
    每个 token 事件格式为 {"stream_type": "token", "node": ..., "text": ...}，
    每个节点完成事件格式为 {"stream_type": "update", "node": ..., "data": ...}。
    """
    agents, initial_state, config = _build_runtime(
        user_input=user_input,
        memory=memory,
        tools=tools,
        chat_model=chat_model,
        thread_id=thread_id,
        verbose=verbose,
        max_iterations=max_iterations,
        uploaded_files=uploaded_files,
    )
    # 全局复用 checkpointer 和 store 实例
    checkpointer = get_checkpointer() if use_postgres_checkpoint else None
    store = get_global_store() if ENABLE_MEMORY else None
    workflow = build_workflow_graph(checkpointer=checkpointer, store=store, **agents)

    active_node: str | None = None

    for event in workflow.stream(initial_state, config=config, stream_mode=stream_mode):
        if isinstance(event, tuple) and len(event) == 2:
            mode_name, event_data = event

            if mode_name == "messages":
                chunk, metadata = event_data
                text = _extract_chunk_text(chunk)
                if not text:
                    continue
                node = _derive_outer_node(metadata, active_node)
                yield {"stream_type": "token", "node": node, "text": text}

            elif mode_name == "updates":
                for node_name, state_update in event_data.items():
                    active_node = node_name
                    yield {
                        "stream_type": "update",
                        "node": node_name,
                        "data": state_update,
                    }
        else:
            # 单 stream mode（如 "updates"）：直接 yield 原始事件，向后兼容
            yield event
