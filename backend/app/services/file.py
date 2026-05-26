"""文件上传与知识库入库服务。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException

from ..doc_asst.rag.ingest import RAGIngestor


class FileService:

    UPLOADS_DIR = Path("workspace") / "uploads"

    def save_upload(self, filename: Optional[str], content: Any, target_dir: Optional[Path] = None) -> Path:
        """将上传文件保存到磁盘，返回保存路径。"""
        save_dir = target_dir or self._ensure_uploads_dir()
        safe_name = os.path.basename(filename or "upload.txt")
        if not safe_name:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        save_path = save_dir / safe_name
        with save_path.open("wb") as f:
            shutil.copyfileobj(content, f)
        return save_path

    def ingest_file(self, file_path: str, tenant_id: str = "default") -> Dict[str, Any]:
        """将文件入库到 RAG 向量库。"""
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        try:
            ingestor = RAGIngestor()
            return ingestor.ingest_file(file_path, tenant_id=tenant_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail="数据库不可用") from exc

    def _ensure_uploads_dir(self) -> Path:
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        return self.UPLOADS_DIR
