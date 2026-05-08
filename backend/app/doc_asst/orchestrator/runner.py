from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..agents import ExtractorSummarizerAgent, PlannerAgent, ReporterAgent
from ..config.settings import DEFAULT_THREAD_ID, ENABLE_MEMORY, ENABLE_POSTGRES_CHECKPOINT, ENABLE_RAG, ENABLE_SEARCH
from ..core.contract import AgentContext, Tool
from ..core.llm import HelloAgentsLLM
from ..persistence.checkpointer import postgres_checkpointer
from ..tools import MemoryTool, RAGTool, SearchTool
from ..tools.registry import ToolRegistry
from .graph import build_workflow_graph
from .state import WorkflowState
from .tool_policy import resolve_tools_for_agent


def run_workflow(
    user_input: str,
    context: Optional[AgentContext] = None,
    memory: Optional[List[Dict[str, str]]] = None,
    tools: Optional[List[Tool]] = None,
    llm: Optional[HelloAgentsLLM] = None,
    thread_id: str = DEFAULT_THREAD_ID,
    use_postgres_checkpoint: bool = ENABLE_POSTGRES_CHECKPOINT,
    verbose: bool = False,
) -> WorkflowState:
    """
    对外统一运行入口。
    可以从 API 层直接调用，后续也可替换为异步/流式版本。
    """

    runtime_context = context or AgentContext()
    # metadata 是跨节点共享的小型“运行时总线”。
    runtime_context.metadata.setdefault("search_query", user_input)
    runtime_context.metadata.setdefault("verbose", verbose)
    runtime_context.metadata.setdefault("thread_id", thread_id)
    runtime_llm = llm or HelloAgentsLLM()
    default_tools: List[Tool] = []
    if ENABLE_SEARCH:
        default_tools.append(SearchTool())
    if ENABLE_RAG:
        default_tools.append(RAGTool())
    if ENABLE_MEMORY:
        default_tools.append(MemoryTool())
    # 允许调用方注入自定义工具（例如测试里的 mock）；未注入时用默认池。
    available_tools: List[Tool] = tools or default_tools
    registry = ToolRegistry(available_tools)

    # 通过策略表分配工具，避免把所有工具暴露给每个 Agent。
    planner_tools = resolve_tools_for_agent("planner_agent", registry)
    extractor_tools = resolve_tools_for_agent("extractor_summarizer_agent", registry)
    reporter_tools = resolve_tools_for_agent("reporter_agent", registry)

    planner = PlannerAgent(llm=runtime_llm, tools=planner_tools)
    extractor = ExtractorSummarizerAgent(llm=runtime_llm, tools=extractor_tools)
    reporter = ReporterAgent(llm=runtime_llm, tools=reporter_tools)

    initial_state: WorkflowState = {
        "user_input": user_input,
        "context": runtime_context,
        "memory": memory,
        "plan_output": {},
        "research_output": {},
        "writer_output": {},
        "plan_text": "",
        "summary_text": "",
        "final_text": "",
        "traces": [],
    }

    # thread_id 会影响 LangGraph 持久化会话的隔离。
    config = {"configurable": {"thread_id": thread_id}}

    if use_postgres_checkpoint:
        with postgres_checkpointer() as checkpointer:
            workflow = build_workflow_graph(
                planner=planner,
                extractor=extractor,
                reporter=reporter,
                checkpointer=checkpointer,
            )
            return workflow.invoke(initial_state, config=config)

    workflow = build_workflow_graph(
        planner=planner,
        extractor=extractor,
        reporter=reporter,
    )
    return workflow.invoke(initial_state, config=config)
