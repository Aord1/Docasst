"""对话工作流服务：同步 / 流式两种模式。"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from fastapi import HTTPException

from ..doc_asst.orchestrator import run_workflow_stream


class ChatService:

    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        max_iterations: int = 2,
        uploaded_files: Optional[List[str]] = None,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """同步执行工作流，返回完整最终状态。"""
        tid = self.resolve_thread_id(thread_id)
        final_state: Dict[str, Any] = {}

        try:
            for chunk in run_workflow_stream(
                user_input=message,
                memory=memory,
                thread_id=tid,
                max_iterations=max_iterations,
                uploaded_files=uploaded_files or None,
            ):
                if isinstance(chunk, dict):
                    for _, value in chunk.items():
                        if isinstance(value, dict):
                            final_state.update(value)
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail="LLM 服务不可用") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM 调用失败: {exc}") from exc

        return {
            "thread_id": tid,
            "final_text": final_state.get("final_text", ""),
            "plan_text": final_state.get("plan_text", ""),
            "summary_text": final_state.get("summary_text", ""),
            "reflection_text": final_state.get("reflection_text", ""),
            "is_simple_query": bool(final_state.get("is_simple_query", False)),
            "token_usage": final_state.get(
                "token_usage",
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        }

    def chat_stream(
        self,
        message: str,
        thread_id: Optional[str] = None,
        max_iterations: int = 2,
        uploaded_files: Optional[List[str]] = None,
        memory: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """流式执行工作流，逐步产出 NDJSON 事件行。"""
        tid = self.resolve_thread_id(thread_id)

        try:
            for chunk in run_workflow_stream(
                user_input=message,
                memory=memory,
                thread_id=tid,
                max_iterations=max_iterations,
                uploaded_files=uploaded_files or None,
            ):
                yield json.dumps(
                    {"type": "chunk", "thread_id": tid, "chunk": chunk},
                    ensure_ascii=False,
                ) + "\n"

            yield json.dumps(
                {"type": "done", "thread_id": tid}, ensure_ascii=False
            ) + "\n"
        except Exception as exc:  # noqa: BLE001
            yield json.dumps(
                {"type": "error", "thread_id": tid, "detail": str(exc)},
                ensure_ascii=False,
            ) + "\n"

    @staticmethod
    def resolve_thread_id(thread_id: Optional[str]) -> str:
        """解析 thread_id：为空则自动生成。"""
        if thread_id and thread_id.strip():
            return thread_id.strip()
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid4().hex[:6]
        return f"session-{now}-{suffix}"
