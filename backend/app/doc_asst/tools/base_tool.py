from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Tools 层的实现基类。
    具体工具只需要实现 _run，公共的异常处理由 run 统一兜底。
    """

    name: str = "base_tool"
    description: str = "Base tool"

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        统一工具返回结构，方便 Agent 侧稳定消费。
        """

        try:
            data = self._run(**kwargs)
            return {
                "ok": True,
                "tool_name": self.name,
                "data": data,
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "tool_name": self.name,
                "data": None,
                "error": str(exc),
            }

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """
        子类实现具体逻辑。
        """

        raise NotImplementedError
