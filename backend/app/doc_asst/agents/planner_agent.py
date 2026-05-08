from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.agent import BaseAgent
from ..core.contract import AgentContext


class PlannerAgent(BaseAgent):
    """
    规划智能体：
    将用户需求拆解为可执行的研究与写作步骤。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是任务规划助手，只负责制定执行计划，不负责给出最终结论。"
        "你必须严格遵守以下规则："
        "1) 只能输出“计划”，禁止直接回答用户问题、禁止给出优缺点分析、禁止给出最终建议。"
        "2) 输出必须包含以下五个部分，并使用这些中文标题："
        "【任务目标】、【执行步骤】、【每步输入】、【每步输出】、【风险与注意事项】。"
        "3) 【执行步骤】至少给出 3 步，且每步使用“步骤1/步骤2/步骤3”格式。"
        "4) 如果用户请求的是分析类问题，也只输出如何分析的计划，不输出分析结果本身。"
        "5) 保持简洁清晰，避免空话。"
        "6) 最后必须追加一个 JSON 代码块，键为: task_goal, steps, step_inputs, step_outputs, risks。"
    )

    def __init__(
        self,
        llm: Any,
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
        context: Optional[AgentContext] = None,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        used_tools: List[str] = []
        search_context = ""
        memory_context = ""
        if "web_search" in self._tool_map:
            search_query = user_input
            if context and context.metadata.get("search_query"):
                search_query = str(context.metadata["search_query"])
            search_result = self.get_tool("web_search").run(query=search_query, max_results=5)
            used_tools.append("web_search")
            if search_result.get("ok"):
                payload = search_result.get("data", {})
                provider = payload.get("provider", "unknown")
                rows = payload.get("results", [])
                lines = []
                for idx, row in enumerate(rows, start=1):
                    lines.append(
                        f"{idx}. {row.get('title', '')}\nURL: {row.get('url', '')}\n摘要: {row.get('snippet', '')}"
                    )
                search_context = (
                    f"已执行联网搜索，数据源: {provider}\n"
                    f"搜索结果:\n{chr(10).join(lines)}\n\n"
                )
            else:
                search_context = f"联网搜索失败: {search_result.get('error')}\n\n"

        if "memory_store" in self._tool_map:
            scope = "default-thread"
            if context and context.metadata.get("thread_id"):
                scope = str(context.metadata["thread_id"])
            recall = self.get_tool("memory_store").run(
                action="recall",
                scope=scope,
                namespace="writer_pref",
                limit=3,
            )
            used_tools.append("memory_store")
            if recall.get("ok"):
                items = recall.get("data", {}).get("items", [])
                if items:
                    memory_context = f"历史写作偏好记忆: {items}\n\n"

        composed_input = f"{memory_context}{search_context}{user_input}"
        messages = self._build_messages(user_input=composed_input, memory=memory)
        plan_text = self._call_llm(messages)
        return {
            "agent": self.name,
            "stage": "plan",
            "plan_text": plan_text,
            "plan_output": {
                "task_goal": user_input,
                "steps": [],
                "step_inputs": [],
                "step_outputs": [],
                "risks": [],
                "raw_plan_text": plan_text,
            },
            "used_tools": used_tools,
            "context": context.metadata if context else {},
        }
