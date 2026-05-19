# DocAsst Architecture

## 1. 项目定位

DocAsst 是一个基于 LangGraph 的多 Agent 文档研究与写作系统，当前支持 CLI、FastAPI、Vue 前端三种入口。

## 2. 目录结构（当前有效）

```text
docasst/
├── pyproject.toml
├── README.md
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/
│       └── doc_asst/
├── frontend/
│   └── src/
├── docs/
└── workspace/
    ├── sources/
    └── uploads/   (运行时创建)
```

后端核心目录：

```text
backend/app/doc_asst/
├── cli.py
├── agents/
├── orchestrator/
├── tools/
├── persistence/
├── rag/
├── core/
└── config/
```

## 3. 执行链路

工作流定义在 `orchestrator/graph.py`：

1. `simple_router`（简单问题直答或进入完整流程）
2. `planner`
3. `extractor_summarizer`
4. `reporter`
5. `reflection`（决定结束或回到 `extractor_summarizer`）

## 4. Agent 职责

- `PlannerAgent`: 拆解任务，输出计划
- `ExtractorSummarizerAgent`: 调工具做事实提取和摘要
- `ReporterAgent`: 生成最终稿，必要时沉淀记忆
- `ReflectionAgent`: 质量评估并控制是否迭代

## 5. 工具层

- `web_search`: Tavily 优先，SerpApi 回退
- `rag_search`: PostgreSQL + pgvector 检索
- `memory_store`: LangGraph Store 的 save/recall
- `file_content_reader`: 读取 txt/md/pdf/docx/图片 OCR

## 6. 持久化

- `persistence/pool.py`: PostgreSQL 连接池单例
- `persistence/checkpointer.py`: LangGraph checkpoint
- `persistence/store.py`: LangGraph store（Postgres 或 InMemory）
- `rag/ingest.py`: 文件切块、向量化、写入 `rag_documents` / `rag_chunks`

## 7. 对外接口

- CLI:
  - `docasst`（主流程）
  - `docasst-ingest`（知识库入库）
- API:
  - `/api/chat`
  - `/api/chat/stream`
  - `/api/files/upload`
  - `/api/knowledge/import`
- Frontend:
  - Vue 3 + Vite 单页应用，默认调用 `http://127.0.0.1:8000`

## 8. 当前边界

- 仍缺少系统化测试基线（路由、工具成功/失败分支）
- RAG 依赖外部 PostgreSQL/pgvector 建表与可用连接
- 部分策略参数仍在代码常量中，待逐步外置到环境变量
