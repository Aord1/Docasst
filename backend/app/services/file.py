"""文件上传与知识库入库服务。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ..doc_asst.rag.ingest import RAGIngestor


class FileService:

    UPLOADS_DIR = Path("workspace") / "uploads"

    def save_upload(self, filename: Optional[str], content: Any, target_dir: Optional[Path] = None) -> Path:
        """将上传文件保存到磁盘，返回保存路径。"""
        save_dir = target_dir or self._ensure_uploads_dir()
        safe_name = os.path.basename(filename or "upload.txt")
        save_path = save_dir / safe_name
        with save_path.open("wb") as f:
            shutil.copyfileobj(content, f)
        return save_path

    def ingest_file(self, file_path: str, tenant_id: str = "default") -> Dict[str, Any]:
        """将文件入库到 RAG 向量库。"""
        ingestor = RAGIngestor()
        return ingestor.ingest_file(file_path, tenant_id=tenant_id)

    def _ensure_uploads_dir(self) -> Path:
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        return self.UPLOADS_DIR
