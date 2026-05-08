from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from ..core.contract import AgentContext


class WorkflowState(TypedDict):
    """
    LangGraph 全局状态。
    所有节点读取并更新这份状态，实现跨节点数据流转。
    """

    user_input: str
    context: AgentContext
    memory: Optional[List[Dict[str, str]]]
    # 三段结构化产物：分别对应 plan / research / write 阶段。
    plan_output: Dict[str, Any]
    research_output: Dict[str, Any]
    writer_output: Dict[str, Any]
    # 三段文本产物：便于直接展示和调试。
    plan_text: str
    summary_text: str
    final_text: str
    # 节点执行轨迹：记录每个 agent 的完整返回。
    traces: List[Dict[str, Any]]
