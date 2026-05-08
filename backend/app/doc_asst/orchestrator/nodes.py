from __future__ import annotations

import json
from typing import Any, Dict

from ..agents import ExtractorSummarizerAgent, PlannerAgent, ReporterAgent
from .state import WorkflowState


def _is_verbose(state: WorkflowState) -> bool:
    """从共享 context 读取日志开关。"""
    context = state.get("context")
    return bool(context and context.metadata.get("verbose"))


def _preview(text: str, limit: int = 160) -> str:
    """日志预览文本，避免终端输出过长。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def planner_node(state: WorkflowState, planner: PlannerAgent) -> Dict[str, Any]:
    """
    规划节点：将用户问题拆解为可执行任务。
    """

    if _is_verbose(state):
        print("[planner] 开始执行")
    result = planner.run(
        user_input=state["user_input"],
        context=state["context"],
        memory=state.get("memory"),
    )
    if _is_verbose(state):
        print(f"[planner] 工具: {result.get('used_tools', [])}")
        print(f"[planner] 输出预览: {_preview(result.get('plan_text', ''))}")
    traces = list(state.get("traces", []))
    traces.append(result)
    # 节点只返回“增量状态”，LangGraph 会与旧状态合并。
    return {
        "plan_text": result.get("plan_text", ""),
        "plan_output": result.get("plan_output", {}),
        "traces": traces,
    }


def extractor_summarizer_node(
    state: WorkflowState,
    extractor: ExtractorSummarizerAgent,
) -> Dict[str, Any]:
    """
    提取总结节点：基于规划结果进行信息提取与摘要。
    """

    if _is_verbose(state):
        print("[extractor] 开始执行")
    composed_input = (
        f"用户原始需求:\n{state['user_input']}\n\n"
        f"规划结果:\n{state.get('plan_text', '')}\n\n"
        "请提取关键信息并输出结构化摘要。"
    )
    result = extractor.run(
        user_input=composed_input,
        context=state["context"],
        memory=state.get("memory"),
    )
    if _is_verbose(state):
        print(f"[extractor] 工具: {result.get('used_tools', [])}")
        print(f"[extractor] 输出预览: {_preview(result.get('summary_text', ''))}")
    traces = list(state.get("traces", []))
    traces.append(result)
    return {
        "summary_text": result.get("summary_text", ""),
        "research_output": result.get("research_output", {}),
        "traces": traces,
    }


def reporter_node(state: WorkflowState, reporter: ReporterAgent) -> Dict[str, Any]:
    """
    结果输出节点：整合规划与摘要，给出最终答复。
    """

    if _is_verbose(state):
        print("[reporter] 开始执行")
    composed_input = (
        f"用户原始需求:\n{state['user_input']}\n\n"
        f"规划结果:\n{state.get('plan_text', '')}\n\n"
        f"规划结构化数据:\n{json.dumps(state.get('plan_output', {}), ensure_ascii=False)}\n\n"
        f"研究摘要:\n{state.get('summary_text', '')}\n\n"
        f"研究结构化数据:\n{json.dumps(state.get('research_output', {}), ensure_ascii=False)}\n\n"
        "请产出最终结果，包含结论与可执行建议。"
    )
    result = reporter.run(
        user_input=composed_input,
        context=state["context"],
        memory=state.get("memory"),
    )
    if _is_verbose(state):
        print(f"[reporter] 工具: {result.get('used_tools', [])}")
        print(f"[reporter] 输出预览: {_preview(result.get('final_text', ''))}")
    traces = list(state.get("traces", []))
    traces.append(result)
    return {
        "final_text": result.get("final_text", ""),
        "writer_output": result.get("writer_output", {}),
        "traces": traces,
    }
