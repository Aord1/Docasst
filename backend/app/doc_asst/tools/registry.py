from __future__ import annotations

from typing import Dict, Iterable, List

from ..core.contract import Tool


class ToolRegistry:
    """
    工具注册中心：
    - 统一保存工具实例
    - 按工具名解析工具列表
    """

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Tool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ValueError(f"未找到工具: {tool_name}") from exc

    def resolve(self, tool_names: List[str]) -> List[Tool]:
        return [self.get(name) for name in tool_names]

    def all_names(self) -> List[str]:
        return list(self._tools.keys())
