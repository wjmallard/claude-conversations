"""Command-line entry points (see [project.scripts] in pyproject.toml)."""

import argparse
import logging
import sys

import psycopg

from claude_conversations import config


def _log():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _ensure_database() -> bool:
    """Create the target database if it doesn't exist. Returns True if created."""
    try:
        psycopg.connect(dbname=config.DB_NAME).close()
        return False
    except psycopg.OperationalError:
        # DB likely missing — create it from the maintenance database.
        with psycopg.connect(dbname="postgres", autocommit=True) as con:
            con.execute(f'CREATE DATABASE "{config.DB_NAME}"')
        return True


def initdb_main():
    ap = argparse.ArgumentParser(description="Create the database and apply the schema")
    ap.add_argument("--reset", action="store_true", help="Drop existing tables first (destroys the index, not your files)")
    args = ap.parse_args()
    _log()

    created = _ensure_database()
    from claude_conversations.db import apply_schema, get_conn
    with get_conn() as conn:
        if args.reset:
            conn.execute("DROP TABLE IF EXISTS messages CASCADE")
            conn.execute("DROP TABLE IF EXISTS conversations CASCADE")
        apply_schema(conn)
    print(f"Database {config.DB_NAME!r} {'created' if created else 'ready'}; schema applied.")
    print("Next: cc-index")


def index_main():
    ap = argparse.ArgumentParser(description="Index conversations into the database")
    ap.add_argument("--reindex", action="store_true", help="Re-process every conversation, ignoring the content-digest cache")
    args = ap.parse_args()
    _log()

    from claude_conversations.db import check_db
    from claude_conversations.indexer import index_archive
    check_db()
    index_archive(reindex=args.reindex)


def embed_main():
    argparse.ArgumentParser(description="Embed messages for semantic search (local MLX)").parse_args()
    _log()

    from claude_conversations.db import check_db
    check_db()
    try:
        from claude_conversations.embedding import backfill_embeddings
    except ImportError as exc:
        print(f"Semantic extras not installed ({exc}).", file=sys.stderr)
        print("Install them with:  uv sync --extra semantic", file=sys.stderr)
        sys.exit(1)
    n = backfill_embeddings()
    print(f"Embedded {n} messages.")


def status_main():
    argparse.ArgumentParser(description="Show index status").parse_args()
    _log()

    from claude_conversations.db import check_db, get_conn, stats
    check_db()
    with get_conn() as conn:
        s = stats(conn)
    print(f"conversations dir : {config.CONVERSATIONS_DIR}")
    print(f"database          : {config.DB_NAME}")
    print(f"conversations     : {s['conversations']}")
    print(f"messages indexed  : {s['messages']}")
    print(f"embedding chunks  : {s['chunks']}")
    print(f"  with embeddings : {s['embedded']}")


def web_main():
    from claude_conversations.web.app import main
    main()
