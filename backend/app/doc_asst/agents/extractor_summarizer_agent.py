from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ..core.agent import BaseAgent


class ExtractorSummarizerAgent(BaseAgent):
    """
    提取与总结智能体：
    接收规划结果与资料内容，提取关键信息并给出结构化总结。
    通过 Function Calling 自主决定调用搜索/RAG/文件读取等工具获取证据。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是研究助手。基于检索证据做事实提取与总结，禁止无依据补充。"
        "输出格式（严格按此，不要加其他标题）：\n"
        "【发现】3~5条关键发现，每条一句。"
        "【缺口】当前缺失的信息，没有则写'无'。"
        "\n\n工具调用策略："
        "\n1. 批量调用：需要多个工具时，在一个响应中同时发起所有 tool_calls。"
        "\n2. 按需调用：已有信息足够时直接输出，不要调用工具。"
        "\n3. 不重复调用：已返回结果的工具不要用相同参数再调。"
        "\n4. 工具选择：外部最新事实→web_search，本地知识库→rag_search，用户文件→file_content_reader。"
    )

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="extractor_summarizer_agent",
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
        summary_text, used_tools, tool_traces, usage = self._run_with_tools(
            user_input=user_input,
            memory=memory,
        )

        evidence = self._extract_evidence_from_traces(tool_traces)

        return {
            "agent": self.name,
            "stage": "research",
            "summary_text": summary_text,
            "evidence": evidence,
            "used_tools": used_tools,
            "tool_traces": tool_traces,
            "usage": usage,
        }

    def _extract_evidence_from_traces(
        self, tool_traces: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """从工具调用追踪中提取结构化证据。"""
        evidence: List[Dict[str, Any]] = []
        for trace in tool_traces:
            tool_name = trace.get("tool_name", "")
            ok = trace.get("ok", False)
            if not ok:
                continue
            evidence.append({
                "source_type": "web" if tool_name == "web_search" else
                               "rag" if tool_name == "rag_search" else
                               "file_upload" if tool_name == "file_content_reader" else
                               "memory" if tool_name == "memory_store" else tool_name,
                "tool_name": tool_name,
                "args": trace.get("args", {}),
            })
        return evidence
