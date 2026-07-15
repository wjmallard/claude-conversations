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
        # DB likely missing -- create it from the maintenance database.
        with psycopg.connect(dbname="postgres", autocommit=True) as con:
            con.execute(f'CREATE DATABASE "{config.DB_NAME}"')
        return True


def _prefer_lz4() -> bool:
    """Ask the database for lz4 TOAST compression, keeping pglz if it isn't available.

    Everything bulky here is TOASTed -- message text, tool I/O, the raw JSON -- and lz4
    is both smaller and several times faster than the pglz default on this data. It is
    set here rather than in schema.sql because a per-column `COMPRESSION lz4` hard-errors
    on a server built without lz4, which would make the schema unusable on a stock build;
    a rejected ALTER just leaves pglz in place. Compression is chosen at write time, so
    this must land before anything is indexed.
    """
    try:
        with psycopg.connect(dbname=config.DB_NAME, autocommit=True) as con:
            con.execute(f'ALTER DATABASE "{config.DB_NAME}" SET default_toast_compression = lz4')
        return True
    except psycopg.Error:
        return False


def initdb_main():
    ap = argparse.ArgumentParser(description="Create the database and apply the schema")
    ap.add_argument("--reset", action="store_true", help="Drop the whole index first, keeping only the cached embeddings")
    args = ap.parse_args()
    _log()

    created = _ensure_database()
    from claude_conversations.db import apply_schema, get_conn
    with get_conn() as conn:
        if args.reset:
            # Drop the entire index and rebuild it empty from the schema. `embeddings`
            # is the one deliberate survivor: it is keyed by content, so re-importing
            # the same prose reuses its vectors and a reset costs no re-embedding.
            # Everything else is cheap to rebuild from the export, so it goes -- a
            # reset should leave nothing behind but the expensive part. Order follows
            # the foreign keys.
            conn.execute("DROP TABLE IF EXISTS export_artifacts CASCADE")
            conn.execute("DROP TABLE IF EXISTS message_chunks CASCADE")
            conn.execute("DROP TABLE IF EXISTS messages CASCADE")
            conn.execute("DROP TABLE IF EXISTS conversations CASCADE")
        apply_schema(conn)
    lz4 = _prefer_lz4()
    print(f"Database {config.DB_NAME!r} {'created' if created else 'ready'}; schema applied.")
    print(f"TOAST compression : {'lz4' if lz4 else 'pglz (lz4 unavailable on this server)'}")
    print("Next: cc-import /path/to/your-export.zip")


def import_main():
    ap = argparse.ArgumentParser(
        description="Import a claude.ai export (.zip) into the database",
    )
    ap.add_argument("export", help="Path to the export .zip (or a bare conversations.json)")
    ap.add_argument("--reimport", action="store_true",
                    help="Rebuild every conversation in the export, even unchanged ones")
    args = ap.parse_args()
    _log()

    from claude_conversations.db import check_db, get_conn
    from claude_conversations.importer import GREW, STALE, import_export

    check_db()
    # Snapshot first: what matters is what this import changed in the DATABASE, and
    # only the before/after can say that.
    with get_conn() as conn:
        before_uuids = {r["uuid"] for r in conn.execute("SELECT uuid FROM conversations").fetchall()}
        before_msgs = conn.execute("SELECT count(*) FROM messages").fetchone()["count"]

    stats = import_export(args.export, reimport=args.reimport)

    with get_conn() as conn:
        after_uuids = conn.execute("SELECT count(*) FROM conversations").fetchone()["count"]
        after_msgs = conn.execute("SELECT count(*) FROM messages").fetchone()["count"]

    statuses = stats["statuses"]
    exported = set(statuses)
    added = exported - before_uuids
    present = exported & before_uuids
    gained = {u for u in present if statuses[u] == GREW}
    stale = {u for u in present if statuses[u] == STALE}
    untouched = len(before_uuids - exported)

    print(f"\nImported into {config.DB_NAME!r}:")
    print(f"  {len(added):6d} conversations new to the database")
    print(f"  {len(present):6d} already present"
          + (f" ({len(gained)} gained new messages)" if present else ""))
    if stale:
        print(f"  {len(stale):6d} older here than what is already indexed (ignored)")
    if untouched:
        print(f"  {untouched:6d} in the database but not in this export (left alone)")
    print(f"\nDatabase now holds {after_uuids} conversations "
          f"({after_uuids - len(before_uuids):+d}) and {after_msgs} messages "
          f"({after_msgs - before_msgs:+d}).")




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
    print(f"database          : {config.DB_NAME}")
    print(f"conversations     : {s['conversations']}")
    print(f"messages indexed  : {s['messages']}")
    print(f"embedding chunks  : {s['chunks']}")
    print(f"  with embeddings : {s['embedded']}")
    print(f"cached vectors    : {s['vectors']}")


def web_main():
    from claude_conversations.web.app import main
    main()
