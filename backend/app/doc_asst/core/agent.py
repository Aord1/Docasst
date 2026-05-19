from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config.settings import MAX_TOOL_ROUNDS


class BaseAgent(ABC):
    """
    Agent 基类：
    - 统一使用 ChatOpenAI 调用 LLM
    - 使用 LangGraph create_react_agent 实现 ReAct 循环
    子类只需实现 run() 编排业务流程。
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: ChatOpenAI,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.2,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.temperature = temperature
        self.tools = tools or []
        self._tool_map = {tool.name: tool for tool in self.tools}
        # P1-7: 缓存 react_agent，避免每次 _run_with_tools 重新编译
        self._react_agent = None
        if self.tools:
            self._react_agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                prompt=self.system_prompt,
            )

    def _build_lc_messages(
        self,
        user_input: str,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> List[BaseMessage]:
        """
        构造 LangChain 消息列表。
        memory 作为历史上下文拼接到 system 与 user 之间。
        """
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        if memory:
            for msg in memory:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=user_input))
        return messages

    def _run_with_tools(
        self,
        user_input: str,
        memory: Optional[List[Dict[str, str]]] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> Tuple[str, List[str], List[Dict[str, Any]], Dict[str, int]]:
        """
        使用 LangGraph create_react_agent 执行 ReAct 循环。
        无工具时直接调用 ChatOpenAI.invoke()。

        Returns:
            (final_text, used_tools, tool_traces)
        """
        lc_messages = self._build_lc_messages(user_input, memory)

        if not self.tools:
            # 无工具，直接调 ChatOpenAI
            response = self.llm.invoke(lc_messages)
            usage = self._extract_usage(response)
            return response.content or "", [], [], usage

        # 使用缓存的 react_agent（__init__ 时已创建）
        result = self._react_agent.invoke(
            {"messages": lc_messages},
            config={"recursion_limit": max_tool_rounds * 2 + 2},
        )

        # 提取结果
        final_text = ""
        used_tools: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_text = msg.content
            if isinstance(msg, AIMessage):
                usage = self._extract_usage(msg)
                usage_total["input_tokens"] += usage["input_tokens"]
                usage_total["output_tokens"] += usage["output_tokens"]
                usage_total["total_tokens"] += usage["total_tokens"]
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name or ""
                used_tools.append(tool_name)
                trace = {
                    "tool_name": tool_name,
                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                    "args": {},
                    "ok": msg.status != "error",
                    "error": msg.content if msg.status == "error" else None,
                }
                tool_traces.append(trace)
                print(f"[{self.name}] 调用工具 {tool_name}({'ok' if trace['ok'] else 'fail'})")

        print(f"\n[{self.name}] ReAct 循环结束，共调用 {len(used_tools)} 次工具")
        return final_text, used_tools, tool_traces, usage_total

    def _extract_usage(self, msg: Any) -> Dict[str, int]:
        """从 LangChain/OpenAI message 中提取 token 使用量。"""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        meta = getattr(msg, "usage_metadata", None) or {}
        if meta:
            usage["input_tokens"] = int(meta.get("input_tokens", 0) or 0)
            usage["output_tokens"] = int(meta.get("output_tokens", 0) or 0)
            usage["total_tokens"] = int(meta.get("total_tokens", 0) or 0)
            return usage
        resp_meta = getattr(msg, "response_metadata", None) or {}
        token_usage = resp_meta.get("token_usage", {}) if isinstance(resp_meta, dict) else {}
        usage["input_tokens"] = int(token_usage.get("prompt_tokens", 0) or 0)
        usage["output_tokens"] = int(token_usage.get("completion_tokens", 0) or 0)
        usage["total_tokens"] = int(token_usage.get("total_tokens", 0) or 0)
        return usage

    @abstractmethod
    def run(
        self,
        user_input: str,
        memory: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        子类必须实现：
        - 调用哪些工具
        - 如何拼接证据
        - 返回什么结构
        """
        raise NotImplementedError
