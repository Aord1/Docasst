# DocAsst

DocAsst 是一个基于 LangGraph 的多 Agent 文档研究与写作助手，支持：

- CLI 对话与流式工作流输出
- FastAPI 后端接口（含流式聊天）
- Vue 3 前端会话界面
- 本地文件读取与知识库（PostgreSQL + pgvector）检索

## 当前项目概览

- 后端：`backend/app`
  - `doc_asst/`：多 Agent 工作流、工具层、RAG、持久化
  - `api/`：HTTP 接口（聊天、文件上传、知识库导入）
  - `main.py`：FastAPI 应用入口
- 前端：`frontend/`
  - Vue 3 + Vite 单页应用
  - 支持会话列表、流式回复、文件上传、知识库导入
- 文档：`docs/`
  - `architecture.md`、`roadmap.md` 等
- 工作目录：`workspace/`
  - `sources/`：知识源文件
  - `uploads/`：上传文件落盘目录（运行时自动创建）

## 工作流（LangGraph）

执行链路：

1. `simple_router`（判断简单问答/完整流程）
2. `planner`
3. `extractor_summarizer`
4. `reporter`
5. `reflection`（可回到 `extractor_summarizer` 继续迭代）

核心工具：

- `web_search`：Tavily 优先，SerpApi 回退
- `rag_search`：PostgreSQL + pgvector 检索
- `memory_store`：记忆存取
- `file_content_reader`：读取 txt/md/pdf/docx/图片 OCR

## 环境要求

- Python `>=3.10`
- Node.js `>=18`（前端开发）
- PostgreSQL（启用 RAG / checkpoint 时建议安装 `pgvector`）

## 快速开始

### 方式一：Docker 部署（推荐）

1. 复制环境变量并填写 API Key：

```bash
cp .env.example .env
# 编辑 .env，至少填写 LLM_API_KEY
```

2. 一键启动：

```bash
docker-compose up -d
```

3. 访问 `http://localhost:8000`

Docker 会自动构建前端、初始化数据库（PostgreSQL + pgvector）、启动后端服务。
上传文件和知识库数据持久化在 Docker volumes 中。

### 方式二：本地开发

#### 1) 安装依赖

推荐使用 `uv`：

```bash
uv sync
```

或使用 `pip`：

```bash
pip install -e .
```

### 2) 配置环境变量

复制并编辑：

```bash
cp .env.example .env
```

关键变量说明：

- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID`：主模型配置
- `VISION_*`：图片 OCR / 视觉模型配置
- `TAVILY_API_KEY`、`SERPAPI_API_KEY`：联网搜索
- `LANGGRAPH_POSTGRES_DSN`：Postgres 连接串
- `EMBEDDING_MODEL_ID`、`EMBEDDING_DIM`：向量化配置

## 运行方式

### A. CLI（最小可用）

```bash
docasst --input "帮我总结 workspace/sources 下的写作规范"
```

常用参数：

- `--thread-id`：会话隔离 ID
- `--max-iterations`：反思迭代次数（默认 2）
- `--file`：上传文件（可重复）
- `--json`：输出完整 JSON
- `--verbose`：显示节点日志

RAG 入库 CLI：

```bash
docasst-ingest --file workspace/sources/writing_guidelines.md --tenant-id default
```

### B. 启动后端 API

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

```bash
GET /api/health
```

核心接口：

- `POST /api/chat`：同步返回最终结果
- `POST /api/chat/stream`：NDJSON 流式输出
- `POST /api/files/upload`：对话附件上传
- `POST /api/knowledge/import`：知识库文件导入并入向量库

### C. 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认前端会请求 `http://127.0.0.1:8000`，请先启动后端。

## 知识库与数据库说明

`RAGIngestor` 默认会写入以下表（需提前建表）：

- `rag_documents`
- `rag_chunks`（`embedding` 字段为 `vector`）

如果未配置可用 Postgres，RAG 与 checkpoint 相关能力将不可用或退化。

## 开发命令参考

- 后端入口：`backend/app/main.py`
- CLI 入口：`backend/app/doc_asst/cli.py`
- 工作流定义：`backend/app/doc_asst/orchestrator/graph.py`
- 前端主界面：`frontend/src/App.vue`

## 已知状态

- API 与前端均已接入，可本地联调
- Roadmap 与后续优化项见 `docs/roadmap.md`
