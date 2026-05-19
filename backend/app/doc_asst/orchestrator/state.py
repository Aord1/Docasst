from __future__ import annotations

from typing import Any, Dict, List, Optional
from langgraph.graph import MessagesState


class WorkflowState(MessagesState):
    """
    LangGraph 全局状态。
    在 MessagesState 基础上扩展业务字段：
    - messages: LangGraph 原生消息状态（由父类提供，自带 reducer 合并）
    - 其它字段: 业务编排状态

    运行时元数据（verbose、uploaded_files 等）通过 LangGraph config 传递，
    不再存储在状态中。
    """

    user_input: str
    memory: Optional[List[Dict[str, str]]]
    # 三段文本产物：便于直接展示和调试。
    plan_text: str
    summary_text: str
    final_text: str
    # 反思阶段产物：用于控制是否继续迭代。
    reflection_text: str
    reflection_output: Dict[str, Any]
    # 简单问题路由标记
    is_simple_query: bool
    simple_query_score: float
    simple_query_reason: str
    # token 统计
    token_usage: Dict[str, int]
    # ReAct 迭代控制字段。
    iteration: int
    max_iterations: int
