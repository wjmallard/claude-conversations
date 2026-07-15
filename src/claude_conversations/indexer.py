"""Scan the conversations directory and (re)build the database index.

Incremental, and split by cost. The .metadata.json sidecar is tiny, so its fields are
refreshed on every run — a renamed or re-summarized conversation lands without touching
the transcript. The expensive half (re-parsing the .jsonl, rebuilding its messages and
embedding chunks) runs only when that file's CONTENT digest changed, or with
reindex=True.

The digest is content-based on purpose: a fresh export rewrites every file with a new
timestamp, so mtime+size bookkeeping would report the entire archive as changed and
drop every embedding through the message_chunks cascade. Both writes for a conversation
share one transaction, so a crash never records a digest for messages that were not
written.

Conversations whose files have disappeared are removed. Raw content is never stored —
only the typed metadata and the provenance-split per-message text (prose + tool_text)
that drives search.
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


_UPSERT_CONV_META = """
    INSERT INTO conversations (
        uuid,
        name,
        summary,
        created_at,
        updated_at,
        account_uuid,
        source_path,
        indexed_at
    )
    VALUES (
        %(uuid)s,
        %(name)s,
        %(summary)s,
        %(created_at)s,
        %(updated_at)s,
        %(account_uuid)s,
        %(source_path)s,
        now()
    )
    ON CONFLICT (uuid) DO UPDATE SET
        name = excluded.name,
        summary = excluded.summary,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at,
        account_uuid = excluded.account_uuid,
        source_path = excluded.source_path,
        indexed_at = now()
"""

# Written only once a transcript's messages have been rebuilt, in the same transaction,
# so an interrupted run never leaves a digest recorded for messages that never landed.
# n_messages counts the whole file, including messages carrying no indexable text.
_UPDATE_CONV_SOURCE = """
    UPDATE conversations
    SET
        n_messages = %(n_messages)s,
        source_sha256 = %(source_sha256)s
    WHERE uuid = %(uuid)s
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
            r["uuid"]: r["source_sha256"]
            for r in conn.execute(
                "SELECT uuid, source_sha256 FROM conversations"
            ).fetchall()
        }

        progress = tqdm(files, desc="Indexing", file=sys.stderr, disable=not verbose)
        for i, (uuid, jsonl_path, meta_path) in enumerate(progress):
            on_disk.append(uuid)
            digest = parse.file_sha256(jsonl_path)

            # The sidecar is cheap to read, so refresh it unconditionally: a rename or
            # an updated summary lands even when the transcript itself is untouched.
            meta = parse.load_metadata(meta_path)
            conn.execute(
                _UPSERT_CONV_META,
                {
                    "uuid": uuid,
                    "name": meta.get("name") or None,
                    "summary": meta.get("summary") or None,
                    "created_at": _parse_ts(meta.get("created_at")),
                    "updated_at": _parse_ts(meta.get("updated_at")),
                    "account_uuid": (meta.get("account") or {}).get("uuid"),
                    "source_path": str(jsonl_path),
                },
            )

            # Everything below re-parses the transcript and re-embeds it; do it only
            # when the bytes actually moved.
            if not reindex and uuid in existing and existing[uuid] == digest:
                skipped += 1
                continue

            messages = parse.load_messages(jsonl_path)
            upload_views = parse.view_upload_names(messages)

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

            conn.execute(
                _UPDATE_CONV_SOURCE,
                {
                    "n_messages": len(messages),
                    "source_sha256": digest,
                    "uuid": uuid,
                },
            )

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
