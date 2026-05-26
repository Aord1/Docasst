from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router as api_router

app = FastAPI(title="DocAsst API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# ── 托管前端静态文件（Docker 部署时生效） ─────────────────
# 前端 build 产物位于 /app/frontend/dist，挂载到 / 根路径
_frontend_dist = Path("/app/frontend/dist")
if _frontend_dist.is_dir():
    # API 路由优先匹配，静态文件兜底
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
