from __future__ import annotations

import argparse
import json

from .config.settings import DEFAULT_THREAD_ID, ENABLE_POSTGRES_CHECKPOINT
from .orchestrator import run_workflow


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = run_workflow(
        user_input=args.input,
        thread_id=args.thread_id,
        use_postgres_checkpoint=args.checkpoint,
        verbose=args.verbose,
    )
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
