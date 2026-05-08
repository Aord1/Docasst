from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.agent import BaseAgent
from ..core.contract import AgentContext


class ExtractorSummarizerAgent(BaseAgent):
    """
    提取与总结智能体：
    接收规划结果与资料内容，提取关键信息并给出结构化总结。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是研究助手（Research Agent）。"
        "你的任务是基于检索证据做事实提取与总结，禁止无依据补充。"
        "输出需包含：关键发现、证据要点、冲突信息、不确定项。"
    )

    def __init__(
        self,
        llm: Any,
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
        context: Optional[AgentContext] = None,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        used_tools: List[str] = []
        evidence: List[Dict[str, Any]] = []
        search_context = ""
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
                snippets = []
                for idx, row in enumerate(rows, start=1):
                    evidence.append(
                        {
                            "source_type": "web",
                            "provider": provider,
                            "title": row.get("title", ""),
                            "source": row.get("url", ""),
                            "snippet": row.get("snippet", ""),
                            "score": None,
                        }
                    )
                    snippets.append(
                        f"{idx}. {row.get('title', '')}\nURL: {row.get('url', '')}\n摘要: {row.get('snippet', '')}"
                    )
                joined = "\n\n".join(snippets)
                search_context = (
                    f"已执行联网搜索，当前使用的数据源: {provider}\n"
                    f"搜索结果如下:\n{joined}\n\n"
                )
            else:
                search_context = f"联网搜索失败: {search_result.get('error')}\n\n"

        rag_context = ""
        if "rag_search" in self._tool_map:
            rag_query = user_input
            if context and context.metadata.get("search_query"):
                rag_query = str(context.metadata["search_query"])
            rag_result = self.get_tool("rag_search").run(query=rag_query, top_k=5)
            used_tools.append("rag_search")
            if rag_result.get("ok"):
                rows = rag_result.get("data", {}).get("results", [])
                lines = []
                for idx, row in enumerate(rows, start=1):
                    evidence.append(
                        {
                            "source_type": "rag",
                            "provider": row.get("source", "local_rag"),
                            "title": row.get("title", ""),
                            "source": row.get("source_path", ""),
                            "snippet": row.get("snippet", ""),
                            "score": row.get("score"),
                        }
                    )
                    lines.append(
                        f"{idx}. {row.get('title', '')}\n来源: {row.get('source_path', '')}\n片段: {row.get('snippet', '')}"
                    )
                rag_context = "本地知识库检索结果:\n" + "\n\n".join(lines) + "\n\n"
            else:
                rag_context = f"本地知识库检索失败: {rag_result.get('error')}\n\n"

        composed_input = f"{search_context}{rag_context}{user_input}"
        messages = self._build_messages(user_input=composed_input, memory=memory)
        summary_text = self._call_llm(messages)
        return {
            "agent": self.name,
            "stage": "research",
            "summary_text": summary_text,
            "research_output": {
                "key_findings": [],
                "conflicts": [],
                "uncertainties": [],
                "evidence": evidence,
                "raw_research_text": summary_text,
            },
            "used_tools": used_tools,
            "context": context.metadata if context else {},
        }
