from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .contract import AgentContext, Tool


class BaseAgent(ABC):
    """
    Agent 基类：
    - 统一消息构造
    - 统一 LLM 调用
    - 统一 Tool 注册/获取
    子类只需实现 run() 编排业务流程。
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: Any,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.2,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.temperature = temperature
        self.tools = tools or []
        self._tool_map = {tool.name: tool for tool in self.tools}

    def _build_messages(
        self,
        user_input: str,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        构造符合 chat.completions 的消息数组。
        memory 作为历史上下文（短期记忆）拼接到 system 与 user 之间。
        """

        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if memory:
            messages.extend(memory)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        统一调用 LLM 客户端。
        当前适配你的 HelloAgentsLLM.think(messages, temperature) 形式。
        """

        result = self.llm.think(messages=messages, temperature=self.temperature)
        if not result:
            raise RuntimeError(f"{self.name} 调用 LLM 失败")
        return result

    def get_tool(self, tool_name: str) -> Tool:
        """
        从注册表中获取工具。
        由子类在 run() 中按需调用，比如 RAGTool、MemoryTool。
        """

        tool = self._tool_map.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")
        return tool

    @abstractmethod
    def run(
        self,
        user_input: str,
        context: Optional[AgentContext] = None,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        子类必须实现：
        - 调用哪些工具
        - 如何拼接证据
        - 返回什么结构
        """

        raise NotImplementedError
