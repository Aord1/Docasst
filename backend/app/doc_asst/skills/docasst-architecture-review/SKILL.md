---
name: docasst-architecture-review
description: Analyze the current project architecture and produce a practical review report with structure, risks, and prioritized improvements. Use when users ask for project structure analysis, architecture review, technical debt assessment, knowledge-point extraction, or refactor recommendations for Python/LangGraph multi-agent projects.
---

# Docasst Architecture Review

## Workflow

1. Map project structure:
- List top-level folders and key entry files.
- Identify runtime entrypoints, orchestration layer, tool layer, and persistence layer.

2. Extract architecture knowledge points:
- Summarize workflow topology (nodes, state, routing).
- Summarize core patterns (ReAct, reflection loop, tool policy, RAG, memory).

3. Detect gaps and risks:
- Flag stale docs that contradict code.
- Flag missing test coverage, weak observability, and configuration hardcoding.
- Flag security or reliability concerns only when backed by code evidence.

4. Produce actionable improvements:
- Group by priority: P0, P1, P2.
- Each item must include scope and expected impact.
- Prefer small, sequenced actions over broad abstract advice.

## Output Contract

Use this exact section order in the final response:

1. `项目结构`
- A concise tree or layered module summary.

2. `核心知识点`
- A short list of patterns and key implementation decisions.

3. `主要问题`
- Findings first, ordered by severity.
- Reference concrete file paths when possible.

4. `改进建议`
- Prioritized P0/P1/P2 action list.

## Guardrails

- Do not claim code facts without reading source files.
- Treat docs as secondary truth; code is primary truth.
- If repository has pre-existing dirty changes, do not revert them.
- Keep recommendations feasible for the current project stage.

## References

When preparing the final review, load:
- `references/review-checklist.md`
