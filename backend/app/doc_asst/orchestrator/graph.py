from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore

from ..agents import ExtractorSummarizerAgent, PlannerAgent, ReflectionAgent, ReporterAgent
from .nodes import (
    extractor_summarizer_node,
    planner_node,
    simple_router_node,
    route_after_simple_router,
    route_after_reporter,
    reflection_node,
    reporter_node,
    route_after_reflection,
)
from .state import WorkflowState


def build_workflow_graph(
    planner: PlannerAgent,
    extractor: ExtractorSummarizerAgent,
    reporter: ReporterAgent,
    reflector: ReflectionAgent,
    checkpointer: Optional[Any] = None,
    store: Optional[BaseStore] = None,
):
    """
    构建 ReAct + Reflection 工作流：
    planner -> extractor_summarizer -> reporter -> reflection -> (extractor_summarizer | END)
    """

    # StateGraph 的泛型是共享状态结构；每个节点读写同一份 state。
    graph = StateGraph(WorkflowState)

    graph.add_node("simple_router", lambda state, config: simple_router_node(state, config))
    graph.add_node("planner", lambda state, config: planner_node(state, planner, config))
    graph.add_node(
        "extractor_summarizer",
        lambda state, config: extractor_summarizer_node(state, extractor, config),
    )
    graph.add_node("reporter", lambda state, config: reporter_node(state, reporter, config))
    graph.add_node("reflection", lambda state, config: reflection_node(state, reflector, config))

    graph.set_entry_point("simple_router")
    graph.add_conditional_edges(
        "simple_router",
        route_after_simple_router,
        {
            "planner": "planner",
            "reporter": "reporter",
        },
    )
    graph.add_edge("planner", "extractor_summarizer")
    graph.add_edge("extractor_summarizer", "reporter")
    graph.add_conditional_edges(
        "reporter",
        route_after_reporter,
        {
            "reflection": "reflection",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "extractor_summarizer": "extractor_summarizer",
            "end": END,
        },
    )

    return graph.compile(checkpointer=checkpointer, store=store)
