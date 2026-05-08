from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, StateGraph

from ..agents import ExtractorSummarizerAgent, PlannerAgent, ReporterAgent
from .nodes import extractor_summarizer_node, planner_node, reporter_node
from .state import WorkflowState


def build_workflow_graph(
    planner: PlannerAgent,
    extractor: ExtractorSummarizerAgent,
    reporter: ReporterAgent,
    checkpointer: Optional[Any] = None,
):
    """
    构建 V1 线性工作流：
    planner -> extractor_summarizer -> reporter -> END
    """

    # StateGraph 的泛型是共享状态结构；每个节点读写同一份 state。
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", lambda state: planner_node(state, planner))
    graph.add_node(
        "extractor_summarizer",
        lambda state: extractor_summarizer_node(state, extractor),
    )
    graph.add_node("reporter", lambda state: reporter_node(state, reporter))

    # V1 采用固定线性流；后续可替换为条件分支或循环（ReAct）。
    graph.set_entry_point("planner")
    graph.add_edge("planner", "extractor_summarizer")
    graph.add_edge("extractor_summarizer", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile(checkpointer=checkpointer)
