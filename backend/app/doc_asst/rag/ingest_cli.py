from __future__ import annotations

import argparse
import json

from .ingest import RAGIngestor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 原文入库脚本")
    parser.add_argument("--file", required=True, help="待入库文件路径")
    parser.add_argument("--tenant-id", default="default", help="租户 ID")
    parser.add_argument(
        "--metadata",
        default="{}",
        help='JSON 字符串，例如: \'{"source":"manual","department":"ops"}\'',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        metadata = json.loads(args.metadata)
        if not isinstance(metadata, dict):
            raise ValueError("metadata 必须是 JSON 对象")
    except Exception as exc:
        raise ValueError(f"--metadata 解析失败: {exc}") from exc

    ingestor = RAGIngestor()
    result = ingestor.ingest_file(
        file_path=args.file,
        tenant_id=args.tenant_id,
        metadata=metadata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
