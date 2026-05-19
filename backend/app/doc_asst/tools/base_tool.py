from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool as LCTool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field, PrivateAttr


class CompactResult(BaseModel):
    """工具精简结果，用于控制返回给 LLM 的 token 量。"""
    ok: bool = True
    tool_name: str = ""
    summary: str = ""
    items: List[Dict[str, Any]] = Field(default_factory=list)


class DocasstBaseTool(LCTool):
    """
    所有 DocAsst 工具的基类。
    继承 LangChain BaseTool，自动兼容 create_react_agent。

    子类需实现:
    - _execute(**kwargs) -> Dict: 执行具体逻辑，返回完整结果 dict
    - compact(result) -> str: 将完整结果精简为字符串，控制 token 量

    返回值:
    - invoke() 返回精简后的字符串（给 LLM 看的）
    - 原始完整结果存在 _last_result 私有属性中（供 traces 使用）

    LangGraph Store 支持:
    - 当图编译时传入 store，工具可通过 self.get_store() 获取
    - 兼容无 store 的场景（返回 None）
    """

    name: str = "base_tool"
    description: str = "Base tool"

    _last_result: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _injected_store: Optional[BaseStore] = PrivateAttr(default=None)

    class _EmptyArgs(BaseModel):
        """默认空参数 schema。"""
        pass

    args_schema: Type[BaseModel] = _EmptyArgs

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result

    def get_store(self) -> Optional[BaseStore]:
        """获取 LangGraph 注入的 Store 实例（如果有的话）。"""
        return self._injected_store

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """
        覆写 invoke：在调用前从 LangGraph config 中捕获 store。
        这样 _execute 内部即可通过 self.get_store() 访问。
        """
        if config:
            store = config.get("store") or config.get("configurable", {}).get("store")
            if store is not None:
                self._injected_store = store
        return super().invoke(input, config=config, **kwargs)

    def _run(self, **kwargs: Any) -> str:
        """
        LangChain BaseTool 接口：返回字符串给 LLM。
        内部调用 _execute 获取完整结果，再通过 compact 精简。
        """
        try:
            result = self._execute(**kwargs)
            self._last_result = {"ok": True, "tool_name": self.name, "data": result, "error": None}
        except Exception as exc:
            self._last_result = {"ok": False, "tool_name": self.name, "data": None, "error": str(exc)}
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        return self.compact(result)

    async def _arun(self, **kwargs: Any) -> str:
        """异步版本（暂用同步实现）。"""
        return self._run(**kwargs)

    def _execute(self, **kwargs: Any) -> Any:
        """子类实现具体逻辑。"""
        raise NotImplementedError

    def compact(self, result: Any) -> str:
        """
        将完整结果精简为字符串返回给 LLM。
        子类可覆写以控制 token 量。
        默认行为：JSON 序列化整个结果。
        """
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False, default=str)
