from __future__ import annotations

import argparse
import json

from .config.settings import DEFAULT_THREAD_ID, ENABLE_POSTGRES_CHECKPOINT
from .orchestrator import run_workflow_stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocAsst CLI")
    parser.add_argument("--input", required=True, help="用户输入内容")
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID, help="LangGraph 线程 ID")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        default=ENABLE_POSTGRES_CHECKPOINT,
        help="启用 PostgreSQL checkpoint 持久化",
    )
    parser.add_argument("--verbose", action="store_true", help="显示节点执行日志与输出预览")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 结果")
    parser.add_argument("--max-iterations", type=int, default=2, help="ReAct + Reflection 最大迭代轮次")
    parser.add_argument("--file", action="append", default=[], help="上传文件路径，可重复传入多个 --file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    final_state = {}
    for chunk in run_workflow_stream(
        user_input=args.input,
        thread_id=args.thread_id,
        use_postgres_checkpoint=args.checkpoint,
        verbose=args.verbose,
        max_iterations=args.max_iterations,
        uploaded_files=args.file or None,
    ):
        print(f"[stream] {json.dumps(chunk, ensure_ascii=False, default=str)}")
        if isinstance(chunk, dict):
            for _, val in chunk.items():
                if isinstance(val, dict):
                    final_state.update(val)
    state = final_state
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return

    print("\n=== 规划 ===")
    print(state.get("plan_text", ""))
    print("\n=== 摘要 ===")
    print(state.get("summary_text", ""))
    print("\n=== 最终结果 ===")
    print(state.get("final_text", ""))


if __name__ == "__main__":
    main()
