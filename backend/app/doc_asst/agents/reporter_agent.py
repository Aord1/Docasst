from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ..core.agent import BaseAgent


class ReporterAgent(BaseAgent):
    """
    结果输出智能体：
    将前序阶段结果整合为最终交付内容（结论、建议、下一步）。
    通过 Function Calling 可自主决定是否将结果存入记忆。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是写作助手。基于上游规划与研究证据输出最终写作结果。"
        "输出格式（严格按此）：\n"
        "【标题】文章标题。\n"
        "【正文】Markdown 格式的正文草稿，800字以内。\n"
        "【来源】引用的关键来源列表。"
        "不要输出提纲或待确认项。"
        "\n\n工具调用策略："
        "\n1. 不要调用 web_search 或 rag_search。"
        "\n2. 仅当内容值得沉淀时调用 memory_store 保存，否则直接输出。"
    )

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="reporter_agent",
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
        final_text, used_tools, tool_traces, usage = self._run_with_tools(
            user_input=user_input,
            memory=memory,
        )
        return {
            "agent": self.name,
            "stage": "write",
            "final_text": final_text,
            "used_tools": used_tools,
            "tool_traces": tool_traces,
            "usage": usage,
        }
