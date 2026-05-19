from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from pydantic import BaseModel, Field, PrivateAttr

from ..core.llm import create_vision_openai_client
from .base_tool import DocasstBaseTool


def _default_allowed_dirs() -> List[str]:
    """默认允许的目录：当前工作目录。"""
    return [os.getcwd()]


class FileContentArgs(BaseModel):
    file_paths: List[str] = Field(default_factory=list, description="要读取的文件路径列表。如未提供，将自动从上下文中获取用户上传的文件路径。")


class FileContentTool(DocasstBaseTool):
    """
    读取上传文件内容，转成可供 LLM 消费的文本证据。
    支持: md/txt/pdf/docx/png/jpg/jpeg/webp/json/csv/rst
    """

    name: str = "file_content_reader"
    description: str = "读取上传文件内容（支持 md/txt/pdf/docx/png/jpg/jpeg/webp/json/csv）并返回文本。用于提取用户上传文件或指定路径文件的内容。"
    args_schema: Type[BaseModel] = FileContentArgs

    max_chars_per_file: int = 8000
    allowed_dirs: List[str] = Field(default_factory=_default_allowed_dirs)
    _client: Any = PrivateAttr(default=None)
    _vision_model: str = PrivateAttr(default="")

    def __init__(self, max_chars_per_file: int = 8000, allowed_dirs: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_chars_per_file = max_chars_per_file
        if allowed_dirs is not None:
            self.allowed_dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        else:
            self.allowed_dirs = [str(Path(d).resolve()) for d in _default_allowed_dirs()]
        client, model = create_vision_openai_client()
        self._client = client
        self._vision_model = model

    def _validate_path(self, raw_path: str) -> Path:
        """校验路径是否在允许的目录内，防止路径遍历攻击。"""
        resolved = Path(raw_path).resolve()
        for allowed in self.allowed_dirs:
            allowed_resolved = Path(allowed).resolve()
            try:
                resolved.relative_to(allowed_resolved)
                return resolved
            except ValueError:
                continue
        raise ValueError(
            f"路径 '{raw_path}' 不在允许的目录内。"
            f"允许的目录: {self.allowed_dirs}"
        )

    def _execute(self, file_paths: List[str] | None = None, **kwargs) -> Dict[str, Any]:
        if not file_paths:
            one = kwargs.get("file_path")
            file_paths = [one] if one else []
        if not file_paths:
            raise ValueError("file_content_reader 需要 file_paths 参数")

        out: List[Dict[str, Any]] = []
        for raw_path in file_paths:
            try:
                path = self._validate_path(str(raw_path))
            except ValueError as exc:
                out.append({"path": str(raw_path), "ok": False, "error": str(exc)})
                continue
            if not path.exists() or not path.is_file():
                out.append({"path": str(path), "ok": False, "error": "文件不存在"})
                continue
            try:
                text = self._extract_text(path)
                text = text[: self.max_chars_per_file]
                out.append({"path": str(path), "ok": True, "suffix": path.suffix.lower(), "text": text})
            except Exception as exc:  # noqa: BLE001
                out.append({"path": str(path), "ok": False, "error": str(exc)})

        return {"files": out}

    def compact(self, result: Any) -> str:
        """精简：每个文件只保留 path + ok + text(1500字)"""
        if isinstance(result, dict) and "files" in result:
            compact_files = []
            for f in result["files"]:
                compact_files.append({
                    "path": f.get("path", ""),
                    "ok": f.get("ok", False),
                    "text": str(f.get("text", ""))[:1500] if f.get("ok") else f.get("error", ""),
                })
            return json.dumps({"ok": True, "files": compact_files}, ensure_ascii=False)
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False, default=str)

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt", ".markdown", ".rst", ".json", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix == ".docx":
            return self._extract_docx(path)
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return self._extract_image_ocr(path)
        if suffix == ".doc":
            raise ValueError("暂不支持 .doc，请先转换为 .docx")
        raise ValueError(f"不支持的文件类型: {suffix}")

    def _extract_pdf(self, path: Path) -> str:
        reader = PdfReader(str(path))
        texts: List[str] = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n\n".join(t.strip() for t in texts if t.strip())

    def _extract_docx(self, path: Path) -> str:
        doc = Document(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(parts)

    def _extract_image_ocr(self, path: Path) -> str:
        mime = "image/png"
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        if path.suffix.lower() == ".webp":
            mime = "image/webp"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        resp = self._client.chat.completions.create(
            model=self._vision_model,
            messages=[
                {"role": "system", "content": "你是OCR助手。请提取图片中的可见文字，原样输出。"},
                {"role": "user", "content": [{"type": "text", "text": "请识别这张图片中的文字内容。"}, {"type": "image_url", "image_url": {"url": data_url}}]},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""
