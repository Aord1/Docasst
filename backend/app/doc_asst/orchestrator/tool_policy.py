from __future__ import annotations

from typing import List

from ..config.settings import AGENT_TOOL_POLICY
from ..core.contract import Tool
from ..tools.registry import ToolRegistry


def resolve_tools_for_agent(agent_name: str, registry: ToolRegistry) -> List[Tool]:
    """根据 settings 中的策略表，解析当前 agent 可用的工具实例。"""
    tool_names = AGENT_TOOL_POLICY.get(agent_name, [])
    return registry.resolve(tool_names)
