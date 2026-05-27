# DocAsst

基于 LangGraph 的多 Agent 文档研究与写作助手，支持联网搜索、知识库检索、文件解析与多轮反思优化。

## 功能特性

- **多 Agent 工作流**：Planner → Extractor/Summarizer → Reporter，支持 Reflection 迭代优化
- **联网搜索**：Tavily 优先，SerpApi 回退
- **RAG 知识库**：PostgreSQL + pgvector 向量检索
- **多格式文件解析**：txt / md / pdf / docx / 图片 OCR
- **流式对话**：NDJSON 实时推送节点进度
- **Vue 3 前端**：会话列表、文件上传、知识库导入

## 项目结构

```
.
├── backend/app/
│   ├── main.py                  # FastAPI 入口
│   ├── api/routes.py            # HTTP 接口定义
│   ├── services/                # 业务服务层（chat、file）
│   └── doc_asst/
│       ├── agents/              # Agent 实现（planner、reporter 等）
│       ├── orchestrator/        # LangGraph 工作流编排
│       ├── tools/               # 工具层（搜索、RAG、文件读取、记忆）
│       ├── rag/                 # RAG 入库与检索
│       ├── skills/              # 技能注册
│       ├── config/              # 配置加载
│       └── cli.py               # CLI 入口
├── frontend/src/
│   └── App.vue                  # Vue 3 单页应用
├── workspace/
│   ├── sources/                 # 知识源文件
│   └── uploads/                 # 上传文件（运行时自动创建）
├── docker-compose.yml           # Docker 一键部署
├── Dockerfile
└── pyproject.toml
```

## 环境要求

- Python >= 3.10
- Node.js >= 18（前端开发）
- PostgreSQL 16+（建议安装 pgvector 扩展）

## 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填写 LLM_API_KEY

# 2. 启动
docker-compose up -d

# 3. 访问
# http://localhost:8000
```

Docker 会自动构建前端、初始化 PostgreSQL + pgvector、启动后端。上传文件和知识库数据持久化在 Docker volumes 中。

### 方式二：本地开发

#### 安装依赖

```bash
# 推荐 uv
uv sync

# 或 pip
pip install -e .
```

#### 配置环境变量

```bash
cp .env.example .env
```

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID` | 主模型（默认 DeepSeek） |
| `VISION_API_KEY` / `VISION_BASE_URL` / `VISION_MODEL_ID` | 视觉/OCR 模型（可选，默认复用 LLM） |
| `TAVILY_API_KEY` / `SERPAPI_API_KEY` | 联网搜索（可选） |
| `LANGGRAPH_POSTGRES_DSN` | PostgreSQL 连接串 |
| `EMBEDDING_MODEL_ID` / `EMBEDDING_DIM` | 向量模型配置 |

#### 启动后端

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

#### 启动前端

```bash
cd frontend && npm install && npm run dev
```

## 使用方式

### CLI

```bash
# 基础用法
docasst --input "帮我总结 workspace/sources 下的写作规范"

# 带文件上传
docasst --input "分析这份报告" --file report.pdf

# 完整参数
docasst --input "..." --thread-id my-session --max-iterations 3 --verbose --json
```

RAG 入库：

```bash
docasst-ingest --file workspace/sources/writing_guidelines.md --tenant-id default
```

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/chat` | POST | 同步对话，返回最终结果 |
| `/api/chat/stream` | POST | NDJSON 流式对话 |
| `/api/files/upload` | POST | 对话附件上传 |
| `/api/knowledge/import` | POST | 知识库导入（入库向量库） |

## 工作流

```
用户输入
  → simple_router（判断简单问答 / 完整流程）
  → planner（拆解任务、规划步骤）
  → extractor_summarizer（搜索、RAG 检索、文件读取、汇总）
  → reporter（生成最终报告）
  → reflection（质量评估，不满意则回到 extractor_summarizer 迭代）
```

## 开发参考

| 模块 | 入口文件 |
|------|---------|
| 后端 API | `backend/app/main.py` |
| CLI | `backend/app/doc_asst/cli.py` |
| 工作流 | `backend/app/doc_asst/orchestrator/graph.py` |
| 前端 | `frontend/src/App.vue` |

详见 `docs/architecture.md` 和 `docs/roadmap.md`。
