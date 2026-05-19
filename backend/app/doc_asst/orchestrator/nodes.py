from __future__ import annotations

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from ..agents import ExtractorSummarizerAgent, PlannerAgent, ReflectionAgent, ReporterAgent
from .state import WorkflowState


def _get_uploaded_files(config: RunnableConfig) -> list:
    """从 LangGraph config 读取上传文件列表。"""
    return config.get("configurable", {}).get("uploaded_files", [])


def _is_verbose(config: RunnableConfig) -> bool:
    """从 LangGraph config 读取日志开关。"""
    return bool(config.get("configurable", {}).get("verbose"))


def _preview(text: str, limit: int = 160) -> str:
    """日志预览文本，避免终端输出过长。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _truncate(text: str, max_chars: int = 2000) -> str:
    """截断过长文本，避免节点间传递臃肿内容撑大 LLM 上下文。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[已截断，原文 {len(text)} 字符]"


def _simple_query_score(user_input: str, uploaded_files: list) -> tuple[bool, float, str]:
    """
    基于多特征打分判断是否为简单问题。
    分数范围近似 [0, 1]，越高越简单。
    """
    text = (user_input or "").strip()
    if not text:
        return True, 1.0, "空输入，按简单问题处理"

    # 起始分：中性
    score = 0.55
    reasons: list[str] = []

    if uploaded_files:
        score -= 0.45
        reasons.append("存在上传文件")

    char_len = len(text)
    if char_len <= 18:
        score += 0.22
        reasons.append("文本很短")
    elif char_len <= 40:
        score += 0.10
        reasons.append("文本较短")
    elif char_len <= 80:
        score -= 0.08
        reasons.append("文本中等长度")
    else:
        score -= 0.25
        reasons.append("文本较长")

    # 简单问答信号
    simple_markers = ["是什么", "谁是", "几点", "多少", "在哪", "怎么读", "翻译", "解释一下", "定义"]
    # 复杂任务信号
    complex_markers = [
        "对比", "分析", "方案", "步骤", "引用", "数据", "research", "总结报告", "联网",
        "调研", "评估", "优化", "架构", "设计", "规划", "路线图", "给出建议", "结合",
    ]
    # 结构化交付信号
    deliverable_markers = ["表格", "markdown", "json", "代码", "清单", "分点", "章节", "报告"]
    # 时效性信号（通常需要检索）
    temporal_markers = ["今天", "最新", "最近", "本周", "今年", "当前", "刚刚", "实时"]
    # 多约束连接词（通常复杂）
    connective_markers = ["并且", "同时", "另外", "然后", "以及", "分别", "先", "再"]

    simple_hits = sum(1 for m in simple_markers if m in text)
    complex_hits = sum(1 for m in complex_markers if m in text)
    deliverable_hits = sum(1 for m in deliverable_markers if m in text)
    temporal_hits = sum(1 for m in temporal_markers if m in text)
    connective_hits = sum(1 for m in connective_markers if m in text)

    if simple_hits:
        score += min(0.18, 0.08 * simple_hits)
        reasons.append(f"简单问答信号+{simple_hits}")
    if complex_hits:
        score -= min(0.36, 0.12 * complex_hits)
        reasons.append(f"复杂任务信号+{complex_hits}")
    if deliverable_hits:
        score -= min(0.24, 0.08 * deliverable_hits)
        reasons.append(f"结构化交付信号+{deliverable_hits}")
    if temporal_hits:
        score -= min(0.18, 0.09 * temporal_hits)
        reasons.append(f"时效性信号+{temporal_hits}")
    if connective_hits >= 2:
        score -= 0.08
        reasons.append("多约束连接词较多")

    # 有多个问号通常不是极简单问答
    q_count = text.count("?") + text.count("？")
    if q_count >= 2:
        score -= 0.10
        reasons.append("多问句")

    # 截断到 [0, 1]
    score = max(0.0, min(1.0, score))
    is_simple = score >= 0.60
    reason = "；".join(reasons) if reasons else "无显著特征"
    return is_simple, score, reason


def _acc_usage(state: WorkflowState, usage: Dict[str, int]) -> Dict[str, int]:
    current = dict(state.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}))
    current["input_tokens"] += int(usage.get("input_tokens", 0))
    current["output_tokens"] += int(usage.get("output_tokens", 0))
    current["total_tokens"] += int(usage.get("total_tokens", 0))
    return current


def simple_router_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    uploaded_files = _get_uploaded_files(config)
    is_simple, score, reason = _simple_query_score(state.get("user_input", ""), uploaded_files)
    return {
        "is_simple_query": is_simple,
        "simple_query_score": score,
        "simple_query_reason": reason,
    }


def route_after_simple_router(state: WorkflowState) -> str:
    if bool(state.get("is_simple_query", False)):
        return "reporter"
    return "planner"


def planner_node(
    state: WorkflowState, planner: PlannerAgent, config: RunnableConfig,
) -> Dict[str, Any]:
    """
    规划节点：将用户问题拆解为可执行任务。
    """
    if _is_verbose(config):
        print("[planner] 开始执行")
    result = planner.run(
        user_input=state["user_input"],
        memory=state.get("memory"),
    )
    if _is_verbose(config):
        print(f"[planner] 工具: {result.get('used_tools', [])}")
        print(f"[planner] 输出预览: {_preview(result.get('plan_text', ''))}")
    return {
        "plan_text": result.get("plan_text", ""),
        "planner_used_tools": result.get("used_tools", []),
        "token_usage": _acc_usage(state, result.get("usage", {})),
    }


def extractor_summarizer_node(
    state: WorkflowState,
    extractor: ExtractorSummarizerAgent,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    提取总结节点：基于规划结果进行信息提取与摘要。
    """
    if _is_verbose(config):
        print("[extractor] 开始执行")
    reflection_feedback = state.get("reflection_output", {})
    feedback_text = ""
    if reflection_feedback:
        feedback_text = (
            f"上一轮反思结论: {reflection_feedback.get('verdict', '')}\n"
            f"缺失点: {reflection_feedback.get('missing_points', [])}\n"
            f"下一步建议: {reflection_feedback.get('next_actions', [])}\n\n"
        )
    composed_input = (
        f"用户原始需求:\n{state['user_input']}\n\n"
        f"规划结果:\n{_truncate(state.get('plan_text', ''), 1500)}\n\n"
        f"{feedback_text}"
    )
    uploaded_files = _get_uploaded_files(config)
    if uploaded_files:
        file_list = ", ".join(uploaded_files)
        composed_input += f"用户上传了以下文件: [{file_list}]，请使用 file_content_reader 工具读取相关文件内容。\n\n"
    composed_input += "请提取关键信息并输出结构化摘要。"
    result = extractor.run(
        user_input=composed_input,
        memory=state.get("memory"),
    )
    if _is_verbose(config):
        print(f"[extractor] 工具: {result.get('used_tools', [])}")
        print(f"[extractor] 输出预览: {_preview(result.get('summary_text', ''))}")
    return {
        "summary_text": result.get("summary_text", ""),
        "extractor_used_tools": result.get("used_tools", []),
        "token_usage": _acc_usage(state, result.get("usage", {})),
    }


def reporter_node(
    state: WorkflowState, reporter: ReporterAgent, config: RunnableConfig,
) -> Dict[str, Any]:
    """
    结果输出节点：整合规划与摘要，给出最终答复。
    """
    if _is_verbose(config):
        print("[reporter] 开始执行")
    reflection_feedback = state.get("reflection_output", {})
    feedback_text = ""
    if reflection_feedback:
        feedback_text = (
            f"上一轮反思结论: {reflection_feedback.get('verdict', '')}\n"
            f"缺失点: {reflection_feedback.get('missing_points', [])}\n"
            f"下一步建议: {reflection_feedback.get('next_actions', [])}\n\n"
        )
    composed_input = (
        f"用户原始需求:\n{state['user_input']}\n\n"
        f"规划要点:\n{_truncate(state.get('plan_text', ''), 800)}\n\n"
        f"研究摘要:\n{_truncate(state.get('summary_text', ''), 2000)}\n\n"
        f"{feedback_text}"
        "请产出最终结果，包含结论与可执行建议。"
    )
    result = reporter.run(
        user_input=composed_input,
        memory=state.get("memory"),
    )
    if _is_verbose(config):
        print(f"[reporter] 工具: {result.get('used_tools', [])}")
        print(f"[reporter] 输出预览: {_preview(result.get('final_text', ''))}")
    return {
        "final_text": result.get("final_text", ""),
        "reporter_used_tools": result.get("used_tools", []),
        "token_usage": _acc_usage(state, result.get("usage", {})),
    }


def reflection_node(
    state: WorkflowState, reflector: ReflectionAgent, config: RunnableConfig,
) -> Dict[str, Any]:
    """
    反思节点：判断答案是否达标，决定是否继续 ReAct 循环。
    """
    if _is_verbose(config):
        print("[reflection] 开始执行")

    composed_input = (
        f"用户原始需求:\n{state['user_input']}\n\n"
        f"当前答案:\n{_truncate(state.get('final_text', ''), 2000)}\n\n"
        "请严格输出 JSON 评估结果。"
    )
    result = reflector.run(
        user_input=composed_input,
        memory=state.get("memory"),
    )
    reflection_output = result.get("reflection_output", {})
    iteration = int(state.get("iteration", 1))
    max_iterations = int(state.get("max_iterations", 2))

    next_iteration = iteration + 1

    if _is_verbose(config):
        verdict = str(reflection_output.get("verdict", "revise")).strip().lower()
        confidence = float(reflection_output.get("confidence", 0.0))
        will_continue = _should_continue(reflection_output, iteration, max_iterations)
        print(f"[reflection] verdict={verdict}, confidence={confidence:.2f}, "
              f"iteration={iteration}/{max_iterations}, continue={will_continue}")
        print(f"[reflection] 输出预览: {_preview(result.get('reflection_text', ''))}")

    return {
        "reflection_text": result.get("reflection_text", ""),
        "reflection_output": reflection_output,
        "iteration": next_iteration,
        "token_usage": _acc_usage(state, result.get("usage", {})),
    }


def route_after_reporter(state: WorkflowState) -> str:
    if bool(state.get("is_simple_query", False)):
        return "end"
    return "reflection"


def _should_continue(
    reflection_output: Dict[str, Any],
    iteration: int,
    max_iterations: int,
) -> bool:
    """判断是否需要继续迭代：未通过 且 未超迭代上限。"""
    verdict = str(reflection_output.get("verdict", "revise")).strip().lower()
    confidence = float(reflection_output.get("confidence", 0.0))
    actually_passed = (verdict == "pass") and (confidence >= 0.7)
    return (not actually_passed) and (iteration <= max_iterations)


def route_after_reflection(state: WorkflowState) -> str:
    """
    条件路由：直接从状态计算是否继续迭代，无需中间字段。
    """
    reflection_output = state.get("reflection_output", {})
    iteration = int(state.get("iteration", 1))
    max_iterations = int(state.get("max_iterations", 2))
    if _should_continue(reflection_output, iteration, max_iterations):
        return "extractor_summarizer"
    return "end"
