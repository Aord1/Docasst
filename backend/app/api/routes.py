from __future__ import annotations

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..doc_asst.orchestrator import run_workflow_stream
from ..doc_asst.rag.ingest import RAGIngestor

router = APIRouter(prefix="/api", tags=["docasst"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User input message")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread id")
    max_iterations: int = Field(default=2, ge=1, le=6)
    uploaded_files: Optional[List[str]] = None
    memory: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    thread_id: str
    final_text: str
    plan_text: str
    summary_text: str
    reflection_text: str
    is_simple_query: bool
    token_usage: Dict[str, int]


def _stream_chat_events(req: ChatRequest, thread_id: str):
    try:
        for chunk in run_workflow_stream(
            user_input=req.message,
            memory=req.memory,
            thread_id=thread_id,
            max_iterations=req.max_iterations,
            uploaded_files=req.uploaded_files or None,
        ):
            payload = {"type": "chunk", "thread_id": thread_id, "chunk": chunk}
            yield json.dumps(payload, ensure_ascii=False) + "\n"

        payload = {"type": "done", "thread_id": thread_id}
        yield json.dumps(payload, ensure_ascii=False) + "\n"
    except Exception as exc:  # noqa: BLE001
        payload = {"type": "error", "thread_id": thread_id, "detail": str(exc)}
        yield json.dumps(payload, ensure_ascii=False) + "\n"


def _save_upload_file(file: UploadFile, target_dir: Path) -> Path:
    file_name = os.path.basename(file.filename or "upload.txt")
    save_path = target_dir / file_name
    with save_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return save_path


def _uploads_dir() -> Path:
    path = Path("workspace") / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collect_final_state(
    user_input: str,
    thread_id: str,
    max_iterations: int,
    uploaded_files: Optional[List[str]],
    memory: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    final_state: Dict[str, Any] = {}
    for chunk in run_workflow_stream(
        user_input=user_input,
        memory=memory,
        thread_id=thread_id,
        max_iterations=max_iterations,
        uploaded_files=uploaded_files or None,
    ):
        if isinstance(chunk, dict):
            for _, value in chunk.items():
                if isinstance(value, dict):
                    final_state.update(value)
    return final_state


def _generate_thread_id() -> str:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:6]
    return f"session-{now}-{suffix}"


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    resolved_thread_id = (req.thread_id or "").strip() or _generate_thread_id()
    try:
        state = _collect_final_state(
            user_input=req.message,
            thread_id=resolved_thread_id,
            max_iterations=req.max_iterations,
            uploaded_files=req.uploaded_files,
            memory=req.memory,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

    return ChatResponse(
        thread_id=resolved_thread_id,
        final_text=state.get("final_text", ""),
        plan_text=state.get("plan_text", ""),
        summary_text=state.get("summary_text", ""),
        reflection_text=state.get("reflection_text", ""),
        is_simple_query=bool(state.get("is_simple_query", False)),
        token_usage=state.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    resolved_thread_id = (req.thread_id or "").strip() or _generate_thread_id()
    return StreamingResponse(
        _stream_chat_events(req, resolved_thread_id),
        media_type="application/x-ndjson; charset=utf-8",
    )


@router.post("/knowledge/import")
async def import_knowledge(
    file: UploadFile = File(...),
    tenant_id: str = Form(default="default"),
) -> Dict[str, Any]:
    uploads_dir = _uploads_dir()
    try:
        save_path = _save_upload_file(file, uploads_dir)
    finally:
        file.file.close()

    try:
        ingestor = RAGIngestor()
        result = ingestor.ingest_file(str(save_path), tenant_id=tenant_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"ingest failed: {exc}") from exc

    return {"ok": True, "saved_path": str(save_path), "ingest": result}


@router.post("/files/upload")
async def upload_chat_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    uploads_dir = _uploads_dir()
    try:
        save_path = _save_upload_file(file, uploads_dir)
    finally:
        file.file.close()
    return {"ok": True, "file_name": file.filename, "saved_path": str(save_path)}
