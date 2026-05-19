# DocAsst Roadmap

## P0（优先）

- 补测试基线：orchestrator 路由 + tools 成功/失败分支
- Agent 输出加结构化校验（Pydantic）
- 统一异常与降级策略（搜索/RAG/LLM）
- 修正依赖与配置基础项（如 `dotenv` -> `python-dotenv`）

## P1（第二阶段）

- 结构化日志（包含 thread_id、tool、latency、status）
- 配置外置化（功能开关与阈值支持环境变量）
- 引用溯源自动化（由 tool traces 自动生成来源）
- API 化（先实现最小可用 `/run` 与流式输出）

## P2（演进）

- ReAct 子图化（提升可观测性）
- 检索质量优化（chunk 策略、重排、metadata filter）
- 前端可视化工作流与调用链
- 成本监控（token、时延、错误率看板）

## 执行规则

- 新增任务先归类到 P0/P1/P2
- 文档只记录“当前状态 + 可执行下一步”
- 超过一个迭代未执行的旧建议要清理或重写
