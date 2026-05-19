from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from ..persistence.store import get_store as get_global_store
from .base_tool import DocasstBaseTool


class MemoryArgs(BaseModel):
    action: str = Field(description="操作类型：save 保存记忆，recall 召回记忆", enum=["save", "recall"])
    memory_type: str = Field(default="short", description="记忆类型：short 短期记忆，long 长期记忆", enum=["short", "long"])
    query: Optional[str] = Field(default=None, description="recall 时的语义检索查询文本")
    key: Optional[str] = Field(default=None, description="save 时的记忆键名")
    value: Optional[Any] = Field(default=None, description="save 时保存的内容")
    limit: int = Field(default=5, description="recall 时返回结果数量上限")
    scope: Optional[str] = Field(default=None, exclude=True)
    namespace: Optional[str] = Field(default="writer", exclude=True)


class MemoryTool(DocasstBaseTool):
    """
    基于 LangGraph Store 的记忆工具（支持"短期 + 长期"）。

    Store 实例来源优先级：
    1. LangGraph config 注入的 store（graph.compile(store=store)）
    2. 全局 get_store() 单例
    """

    name: str = "memory_store"
    description: str = "持久化记忆工具，支持 save（保存）和 recall（召回）。短期记忆按会话隔离，长期记忆跨会话持久化。"
    args_schema: Type[BaseModel] = MemoryArgs

    def _get_store(self) -> BaseStore:
        """
        获取 Store 实例。
        优先使用 LangGraph 注入的 store，否则 fallback 到全局单例。
        """
        store = self.get_store()
        if store is not None:
            return store
        return get_global_store()

    def _execute(self, action: str = "", memory_type: str = "short", query: Optional[str] = None, key: Optional[str] = None, value: Any = None, limit: int = 5, scope: Optional[str] = None, namespace: str = "writer", **kwargs) -> Dict[str, Any]:
        action = action.strip().lower()
        memory_type = memory_type.strip().lower()
        if memory_type not in {"short", "long"}:
            raise ValueError("memory_type 仅支持 short 或 long")
        if not scope:
            scope = "default-thread" if memory_type == "short" else "default-user"

        store = self._get_store()

        if action == "save":
            if not key:
                raise ValueError("save 动作需要 key")
            self._save(store, memory_type, scope, namespace, key, value)
            return {"action": "save", "memory_type": memory_type, "scope": scope, "namespace": namespace, "key": key, "saved": True}

        if action == "recall":
            rows = self._recall(store, memory_type, scope, namespace, key, query, limit)
            return {"action": "recall", "memory_type": memory_type, "scope": scope, "namespace": namespace, "query": query, "items": rows}

        raise ValueError("memory_store action 仅支持 save 或 recall")

    def compact(self, result: Any) -> str:
        """记忆结果精简：只保留核心字段"""
        if isinstance(result, dict):
            compact = {"ok": True, "action": result.get("action")}
            if result.get("action") == "save":
                compact["key"] = result.get("key", "")
                compact["saved"] = True
            elif result.get("action") == "recall":
                items = result.get("items", [])[:3]
                compact["items"] = [{"key": i.get("key"), "value": i.get("value")} for i in items]
            return json.dumps(compact, ensure_ascii=False, default=str)
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False, default=str)

    def _save(self, store: BaseStore, memory_type: str, scope: str, namespace: str, key: str, value: Any) -> None:
        value_json = value if isinstance(value, dict) else {"raw": value}
        text = json.dumps(value_json, ensure_ascii=False)
        store.put(
            self._namespace_tuple(memory_type, scope, namespace),
            key,
            {"memory_type": memory_type, "scope": scope, "namespace": namespace, "key": key, "text": text, "value": value_json},
            index=["text"],
        )

    def _recall(self, store: BaseStore, memory_type: str, scope: str, namespace: str, key: str | None, query: str | None, limit: int) -> List[Dict[str, Any]]:
        ns = self._namespace_tuple(memory_type, scope, namespace)
        if key:
            item = store.get(ns, key)
            if not item:
                return []
            return [{"key": item.key, "value": item.value.get("value"), "score": None, "updated_at": item.updated_at.isoformat() if item.updated_at else None}]
        results = store.search(ns, query=query, limit=limit)
        return [{"key": it.key, "value": it.value.get("value"), "score": it.score, "updated_at": it.updated_at.isoformat() if it.updated_at else None} for it in results]

    def _namespace_tuple(self, memory_type: str, scope: str, namespace: str) -> tuple[str, ...]:
        return ("docasst_memory", memory_type, scope, namespace)
