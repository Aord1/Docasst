from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def _resolve_tools_for_agent(agent_name: str, available_tools: List) -> List:
    """根据 agent 名称从可用工具列表中解析所需工具。"""
    from ..config.settings import AGENT_TOOL_POLICY
    tool_names = AGENT_TOOL_POLICY.get(agent_name, [])
    tool_map = {t.name: t for t in available_tools}
    return [tool_map[name] for name in tool_names if name in tool_map]


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
    stream_mode: str = "updates",
    uploaded_files: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    LangGraph 流式执行：逐步产出节点更新事件。
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
    for chunk in workflow.stream(initial_state, config=config, stream_mode=stream_mode):
        yield chunk
