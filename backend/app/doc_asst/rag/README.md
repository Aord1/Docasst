# RAG V1 说明

当前 `RAGTool` 是轻量检索版本，适合 CLI 第一版快速跑通：

1. 数据目录：`workspace/sources`（可用环境变量 `RAG_SOURCE_DIR` 覆盖）
2. 支持格式：`.md`、`.txt`
3. 策略：分段 -> 关键词重叠召回 -> 返回 top_k

## 未来升级路径（向量数据库）

1. 在 `rag/ingest.py` 加入 embedding 入库
2. 在 `rag/vectorstore.py` 抽象向量库适配层
3. 把 `RAGTool._rank` 替换为向量检索 + 重排
4. 默认仍返回相同结构，保证 Agent 侧无需改动
