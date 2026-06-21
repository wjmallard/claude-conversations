"""Scan the conversations directory and (re)build the database index.

Incremental: a conversation is re-processed only when its .jsonl mtime or size
changed since last index (or with reindex=True). Conversations whose files have
disappeared are removed. Raw content is never stored — only the typed metadata
and the provenance-split per-message text (prose + tool_text) that drives search.
"""

import sys
from datetime import datetime

from tqdm import tqdm

from claude_conversations import categories, config, parse
from claude_conversations.db import get_conn


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


_UPSERT_CONV = """
    INSERT INTO conversations (
        uuid,
        name,
        summary,
        created_at,
        updated_at,
        account_uuid,
        n_messages,
        source_path,
        source_mtime,
        source_size,
        indexed_at
    )
    VALUES (
        %(uuid)s,
        %(name)s,
        %(summary)s,
        %(created_at)s,
        %(updated_at)s,
        %(account_uuid)s,
        %(n_messages)s,
        %(source_path)s,
        %(source_mtime)s,
        %(source_size)s,
        now()
    )
    ON CONFLICT (uuid) DO UPDATE SET
        name = excluded.name,
        summary = excluded.summary,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at,
        account_uuid = excluded.account_uuid,
        n_messages = excluded.n_messages,
        source_path = excluded.source_path,
        source_mtime = excluded.source_mtime,
        source_size = excluded.source_size,
        indexed_at = now()
"""

_INSERT_MSG = """
    INSERT INTO messages (
        uuid,
        conv_uuid,
        seq,
        sender,
        created_at,
        text,
        tool_text
    )
    VALUES (
        %(uuid)s,
        %(conv_uuid)s,
        %(seq)s,
        %(sender)s,
        %(created_at)s,
        %(text)s,
        %(tool_text)s
    )
    ON CONFLICT (uuid) DO NOTHING
"""

_INSERT_CHUNK = """
    INSERT INTO message_chunks (
        msg_uuid,
        conv_uuid,
        seq,
        sender,
        created_at,
        text
    )
    VALUES (
        %(msg_uuid)s,
        %(conv_uuid)s,
        %(seq)s,
        %(sender)s,
        %(created_at)s,
        %(text)s
    )
"""


def index_archive(reindex=False, verbose=True):
    """(Re)index every conversation in config.CONVERSATIONS_DIR. Returns counts dict."""
    files = list(parse.iter_conversation_files(config.CONVERSATIONS_DIR))
    if verbose:
        print(f"Scanning {len(files)} conversations in {config.CONVERSATIONS_DIR}", file=sys.stderr)

    changed = skipped = msg_total = 0
    on_disk = []

    with get_conn() as conn:
        existing = {
            r["uuid"]: (r["source_mtime"], r["source_size"])
            for r in conn.execute(
                "SELECT uuid, source_mtime, source_size FROM conversations"
            ).fetchall()
        }

        progress = tqdm(files, desc="Indexing", file=sys.stderr, disable=not verbose)
        for i, (uuid, jsonl_path, meta_path) in enumerate(progress):
            on_disk.append(uuid)
            mtime, size = parse.file_stat(jsonl_path)

            if not reindex and uuid in existing:
                old_mtime, old_size = existing[uuid]
                if old_mtime is not None and abs((old_mtime or 0) - mtime) < 1e-6 and old_size == size:
                    skipped += 1
                    continue

            meta = parse.load_metadata(meta_path)
            messages = parse.load_messages(jsonl_path)
            upload_views = parse.view_upload_names(messages)

            conn.execute(
                _UPSERT_CONV,
                {
                    "uuid": uuid,
                    "name": meta.get("name") or None,
                    "summary": meta.get("summary") or None,
                    "created_at": _parse_ts(meta.get("created_at")),
                    "updated_at": _parse_ts(meta.get("updated_at")),
                    "account_uuid": (meta.get("account") or {}).get("uuid"),
                    "n_messages": len(messages),
                    "source_path": str(jsonl_path),
                    "source_mtime": mtime,
                    "source_size": size,
                },
            )

            # Replace this conversation's searchable messages.
            conn.execute(
                "DELETE FROM messages WHERE conv_uuid = %(uuid)s",
                {"uuid": uuid},
            )
            rows, chunk_rows = [], []
            for seq, msg in enumerate(messages):
                prose, tool = parse.message_texts(msg, upload_views)
                if not prose and not tool:
                    continue
                muuid = msg.get("uuid") or f"{uuid}:{seq}"
                created = _parse_ts(msg.get("created_at"))
                sender = msg.get("sender")
                rows.append({
                    "uuid": muuid,
                    "conv_uuid": uuid,
                    "seq": seq,
                    "sender": sender,
                    "created_at": created,
                    "text": prose,
                    "tool_text": tool or None,
                })
                # Only prose is embedded. A short message yields one chunk; a long
                # pasted document yields several (so semantic search covers it all).
                for cseq, chunk in enumerate(parse.chunk_text(prose)):
                    chunk_rows.append({
                        "msg_uuid": muuid,
                        "conv_uuid": uuid,
                        "seq": cseq,
                        "sender": sender,
                        "created_at": created,
                        "text": chunk,
                    })
            if rows:
                conn.cursor().executemany(_INSERT_MSG, rows)
                msg_total += len(rows)
            if chunk_rows:
                conn.cursor().executemany(_INSERT_CHUNK, chunk_rows)

            changed += 1
            if (i + 1) % 200 == 0:
                conn.commit()

        # Remove conversations whose files are gone.
        removed = 0
        if existing:
            gone = set(existing) - set(on_disk)
            if gone:
                conn.execute(
                    "DELETE FROM conversations WHERE uuid = ANY(%(gone)s)",
                    {"gone": list(gone)},
                )
                removed = len(gone)

        # Re-apply category tags from the curation file (rebuildable index).
        # Tags are auxiliary — never fail an index over them.
        try:
            categories.sync_to_db(conn)
        except Exception as exc:
            print(f"Warning: category sync skipped ({exc})", file=sys.stderr)

    if verbose:
        print(
            f"Done: {changed} (re)indexed, {skipped} unchanged, {removed} removed; "
            f"{msg_total} searchable messages written.",
            file=sys.stderr,
        )
    return {
        "changed": changed,
        "skipped": skipped,
        "removed": removed,
        "messages": msg_total,
    }
