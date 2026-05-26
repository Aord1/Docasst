# ── Stage 1: 构建前端 ──────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: 构建后端 + 运行时 ─────────────────────────────
FROM python:3.12-slim

# 系统依赖（psycopg[binary] 需要 libpq）
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制项目元数据和锁文件，安装依赖（利用 Docker 缓存层）
COPY pyproject.toml uv.lock ./

# 安装 uv → 同步依赖 → 安装项目本体 → 清理 uv
COPY backend/ backend/
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev && \
    pip uninstall -y uv

# 从前端构建阶段复制产物
COPY --from=frontend-build /build/frontend/dist /app/frontend/dist

# 文档（CLI 引用等）
COPY docs/ docs/

# 创建工作目录
RUN mkdir -p /app/workspace/sources /app/workspace/uploads

# 环境变量默认值
ENV PYTHONUNBUFFERED=1 \
    LANGGRAPH_POSTGRES_DSN="" \
    LLM_API_KEY="" \
    LLM_BASE_URL="" \
    LLM_MODEL_ID="" \
    VISION_API_KEY="" \
    VISION_BASE_URL="" \
    VISION_MODEL_ID="" \
    TAVILY_API_KEY="" \
    SERPAPI_API_KEY="" \
    EMBEDDING_MODEL_ID=text-embedding-3-small \
    EMBEDDING_DIM=1536 \
    RAG_SOURCE_DIR=workspace/sources \
    PG_POOL_MIN_SIZE=2 \
    PG_POOL_MAX_SIZE=10

EXPOSE 8000

# 入口脚本：等 PG 就绪 → 建表 → 启动服务
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
