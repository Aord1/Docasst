from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ..core.agent import BaseAgent


class PlannerAgent(BaseAgent):
    """
    规划智能体：
    将用户需求拆解为可执行的研究与写作步骤。
    通过 Function Calling 自主决定是否调用搜索/RAG/文件读取等工具获取背景信息。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是任务规划助手，只输出计划，不输出结论。"
        "规则："
        "1) 禁止回答用户问题、禁止给出分析或建议。"
        "2) 输出格式（严格按此，不要加其他标题）：\n"
        "【目标】一句话描述任务目标。\n"
        "【步骤】2~3步，每步一行，格式：步骤N: 简述。\n"
        "【风险】1~2个关键风险，没有则写'无'。"
        "3) 每步描述不超过30字。不要输出JSON。"
        "\n\n工具调用策略："
        "你只有 web_search 可用。仅在对话题完全不熟悉时调用一次。"
        "调用后立即输出计划，不再调用工具。"
        "如果已理解需求，直接输出计划。"
    )

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="planner_agent",
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            llm=llm,
            tools=tools,
            temperature=temperature,
        )

    def run(
        self,
        user_input: str,
        memory: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        plan_text, used_tools, tool_traces, usage = self._run_with_tools(
            user_input=user_input,
            memory=memory,
        )
        return {
            "agent": self.name,
            "stage": "plan",
            "plan_text": plan_text,
            "used_tools": used_tools,
            "tool_traces": tool_traces,
            "usage": usage,
        }
