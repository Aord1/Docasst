from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.agent import BaseAgent
from ..core.contract import AgentContext


class ReporterAgent(BaseAgent):
    """
    结果输出智能体：
    将前序阶段结果整合为最终交付内容（结论、建议、下一步）。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是写作助手（Writer Agent）。"
        "请基于上游规划与研究证据输出可直接使用的写作结果。"
        "输出必须包含：标题、提纲、正文草稿、引用来源、待确认项。"
    )

    def __init__(
        self,
        llm: Any,
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
        context: Optional[AgentContext] = None,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        used_tools: List[str] = []
        messages = self._build_messages(user_input=user_input, memory=memory)
        final_text = self._call_llm(messages)

        if "memory_store" in self._tool_map:
            scope = "default-thread"
            if context and context.metadata.get("thread_id"):
                scope = str(context.metadata["thread_id"])
            save_res = self.get_tool("memory_store").run(
                action="save",
                scope=scope,
                namespace="writer_pref",
                key="last_output_summary",
                value={"preview": final_text[:280]},
            )
            used_tools.append("memory_store")
            if not save_res.get("ok"):
                # 记忆失败不阻断写作主流程
                pass

        return {
            "agent": self.name,
            "stage": "write",
            "final_text": final_text,
            "writer_output": {
                "title": "",
                "outline": [],
                "draft_markdown": final_text,
                "citations": [],
                "uncertainties": [],
            },
            "used_tools": used_tools,
            "context": context.metadata if context else {},
        }
