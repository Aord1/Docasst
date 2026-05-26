#!/usr/bin/env bash
set -euo pipefail

echo "=== DocAsst Entrypoint ==="

# ── 1. 等待 PostgreSQL 就绪 ────────────────────────────────
if [ -n "${LANGGRAPH_POSTGRES_DSN:-}" ]; then
    echo "[entrypoint] 等待 PostgreSQL 就绪..."
    for i in $(seq 1 30); do
        if python -c "
import psycopg
conn = psycopg.connect('${LANGGRAPH_POSTGRES_DSN}')
conn.close()
" 2>/dev/null; then
            echo "[entrypoint] PostgreSQL 已就绪"
            break
        fi
        echo "[entrypoint] 等待中... ($i/30)"
        sleep 2
    done
fi

# ── 2. 初始化数据库表 ──────────────────────────────────────
if [ -n "${LANGGRAPH_POSTGRES_DSN:-}" ]; then
    echo "[entrypoint] 初始化数据库表..."
    python -c "
import psycopg

dsn = '${LANGGRAPH_POSTGRES_DSN}'

with psycopg.connect(dsn) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # 启用 pgvector 扩展
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        print('  ✓ pgvector 扩展已启用')

        # rag_documents 表
        cur.execute('''
            CREATE TABLE IF NOT EXISTS rag_documents (
                doc_id        TEXT PRIMARY KEY,
                tenant_id     TEXT NOT NULL DEFAULT '\''default'\'',
                source_path   TEXT NOT NULL,
                title         TEXT NOT NULL,
                content       TEXT NOT NULL,
                content_hash  TEXT NOT NULL,
                version       INTEGER NOT NULL DEFAULT 1,
                metadata      JSONB NOT NULL DEFAULT '\''{}'\'',
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        ''')
        print('  ✓ rag_documents 表已就绪')

        # rag_chunks 表
        cur.execute('''
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id      TEXT PRIMARY KEY,
                doc_id        TEXT NOT NULL REFERENCES rag_documents(doc_id) ON DELETE CASCADE,
                tenant_id     TEXT NOT NULL DEFAULT '\''default'\'',
                source_path   TEXT NOT NULL,
                title         TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                text         TEXT NOT NULL,
                embedding    vector(${EMBEDDING_DIM:-1536}),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        ''')
        print('  ✓ rag_chunks 表已就绪')

        # 索引
        cur.execute('CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc_id ON rag_chunks(doc_id)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_rag_documents_tenant ON rag_documents(tenant_id)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_rag_chunks_tenant ON rag_chunks(tenant_id)')
        print('  ✓ 索引已创建')
"
    echo "[entrypoint] 数据库初始化完成"
fi

# ── 3. 启动应用 ────────────────────────────────────────────
echo "[entrypoint] 启动 DocAsst..."
exec "$@"
