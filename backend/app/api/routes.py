from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..services import ChatService, FileService

router = APIRouter(prefix="/api", tags=["docasst"])

# ── 服务实例 ────────────────────────────────────────────────
_chat_svc = ChatService()
_file_svc = FileService()


# ── 请求 / 响应模型 ────────────────────────────────────────
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


# ── 端点 ───────────────────────────────────────────────────
@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = _chat_svc.chat(
        message=req.message,
        thread_id=req.thread_id,
        max_iterations=req.max_iterations,
        uploaded_files=req.uploaded_files,
        memory=req.memory,
    )
    return ChatResponse(**result)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    event_iter = _chat_svc.chat_stream(
        message=req.message,
        thread_id=req.thread_id,
        max_iterations=req.max_iterations,
        uploaded_files=req.uploaded_files,
        memory=req.memory,
    )
    return StreamingResponse(
        event_iter,
        media_type="application/x-ndjson; charset=utf-8",
    )


@router.post("/knowledge/import")
async def import_knowledge(
    file: UploadFile = File(...),
    tenant_id: str = Form(default="default"),
) -> Dict[str, Any]:
    try:
        save_path = _file_svc.save_upload(file.filename, file.file)
    finally:
        file.file.close()

    result = _file_svc.ingest_file(str(save_path), tenant_id=tenant_id)
    return {"ok": True, "saved_path": str(save_path), "ingest": result}


@router.post("/files/upload")
async def upload_chat_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        save_path = _file_svc.save_upload(file.filename, file.file)
    finally:
        file.file.close()
    return {"ok": True, "file_name": file.filename, "saved_path": str(save_path)}
