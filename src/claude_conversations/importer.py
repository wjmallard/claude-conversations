"""Import a claude.ai data export into the database.

An export is a .zip holding conversations.json -- a JSON array of conversations, each
carrying its messages inline under `chat_messages` -- plus a few small side files
(users, memories, projects, reflections). It reads straight out of the zip in about a
second: no extraction, no staging directory, no external tooling.

Everything in the zip lands in Postgres and nothing is read back from disk. The export
.zip is the source of truth; the database is a rebuildable index over it, holding each
message exactly as exported (messages.raw) alongside the typed columns and split text
that drive search. There is deliberately no intermediate copy of the archive: one used
to exist, and its only lasting effect was drifting out of sync with the export it came
from and making "drop the database" mean less than it says.

Importing MERGES, and never deletes. claude.ai issues incremental exports covering
only a recent window, so an export is usually a slice of the archive rather than a
snapshot of it: a conversation this export does not mention is left exactly as it is.
That also means a conversation deleted upstream is not deleted here -- an incremental
cannot distinguish "deleted" from "outside the window", so it does not guess.

A conversation whose messages are byte-identical to what is already indexed is skipped
entirely: not re-parsed, not rebuilt, and -- because rebuilding cascades chunks away --
its cached embeddings are never disturbed.
"""

import sys
import zipfile
from datetime import datetime
from pathlib import Path

import orjson
from psycopg.types.json import Jsonb
from tqdm import tqdm

from claude_conversations import categories, config, parse
from claude_conversations.db import get_conn

_CONVERSATIONS_MEMBER = "conversations.json"

# Per-conversation outcomes, reported against the DATABASE by cc-import.
NEW = "new"              # not indexed before
GREW = "grew"            # indexed, and the export carries messages we lack
CHANGED = "changed"      # indexed with the same messages, but their content differs
UNCHANGED = "unchanged"  # byte-for-byte what is already indexed
STALE = "stale"          # this export predates what is indexed; ignored


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


# Never let an older export overwrite a newer one. Exports overlap -- a full export and
# a 90-day incremental share a window -- so importing them in an arbitrary order must be
# safe. The WHERE guard makes the upsert a no-op when the incoming snapshot predates
# what is indexed, so import order stops mattering.
_UPSERT_CONV_META = """
    INSERT INTO conversations (
        uuid,
        name,
        summary,
        created_at,
        updated_at,
        account_uuid,
        indexed_at
    )
    VALUES (
        %(uuid)s,
        %(name)s,
        %(summary)s,
        %(created_at)s,
        %(updated_at)s,
        %(account_uuid)s,
        now()
    )
    ON CONFLICT (uuid) DO UPDATE SET
        name = excluded.name,
        summary = excluded.summary,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at,
        account_uuid = excluded.account_uuid,
        indexed_at = now()
    WHERE conversations.updated_at IS NULL
       OR excluded.updated_at IS NULL
       OR excluded.updated_at >= conversations.updated_at
"""

# Written only once a conversation's messages have been rebuilt, in the same
# transaction, so an interrupted import never records a digest for messages that never
# landed. n_messages counts the whole transcript, including messages with no indexable
# text.
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
        parent_uuid,
        sender,
        created_at,
        text,
        tool_text,
        raw
    )
    VALUES (
        %(uuid)s,
        %(conv_uuid)s,
        %(seq)s,
        %(parent_uuid)s,
        %(sender)s,
        %(created_at)s,
        %(text)s,
        %(tool_text)s,
        %(raw)s
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
        text,
        text_sha256
    )
    VALUES (
        %(msg_uuid)s,
        %(conv_uuid)s,
        %(seq)s,
        %(sender)s,
        %(created_at)s,
        %(text)s,
        %(text_sha256)s
    )
"""

_UPSERT_ARTIFACT = """
    INSERT INTO export_artifacts (
        kind,
        uuid,
        raw,
        imported_at
    )
    VALUES (
        %(kind)s,
        %(uuid)s,
        %(raw)s,
        now()
    )
    ON CONFLICT (kind, uuid) DO UPDATE SET
        raw = excluded.raw,
        imported_at = now()
"""


def read_export(path) -> list[dict]:
    """Return the conversation list from an export .zip, or a bare conversations.json."""
    path = Path(path).expanduser()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if Path(n).name == _CONVERSATIONS_MEMBER]
            if not members:
                raise ValueError(f"{path}: archive contains no {_CONVERSATIONS_MEMBER}")
            blob = z.read(members[0])
    else:
        blob = path.read_bytes()
    convs = orjson.loads(blob)
    if not isinstance(convs, list):
        raise ValueError(f"{path}: expected a JSON array of conversations")
    return convs


def read_artifacts(path) -> list[tuple]:
    """Yield (kind, uuid, obj) for every non-conversation JSON member of the export.

    users.json and memories.json ship one object each (uuid ''); projects/ and
    reflections/ ship one per uuid. Nothing reads these yet -- they are stored verbatim
    because they are already in the file, and dropping them would mean re-importing to
    get them back.
    """
    path = Path(path).expanduser()
    if not zipfile.is_zipfile(path):
        return []
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            p = Path(name)
            if p.name == _CONVERSATIONS_MEMBER or p.suffix != ".json":
                continue
            try:
                obj = orjson.loads(z.read(name))
            except orjson.JSONDecodeError:
                continue
            if p.parent.name:            # projects/<uuid>.json
                out.append((p.parent.name, p.stem, obj))
            else:                        # users.json, memories.json
                out.append((p.stem, "", obj))
    return out


def import_export(path, reimport=False, verbose=True) -> dict:
    """Import an export .zip into the database. Returns {conversations, messages, statuses}.

    `statuses` maps each conversation uuid in the export to NEW/GREW/CHANGED/UNCHANGED,
    measured against what was already indexed. With reimport=True every conversation is
    rebuilt even when its digest matches (embeddings still survive: they are cached by
    text, not owned by a chunk row).
    """
    convs = read_export(path)
    n_msgs = sum(len(c.get("chat_messages") or []) for c in convs)
    if verbose:
        print(f"Export holds {len(convs)} conversations, {n_msgs} messages", file=sys.stderr)

    statuses = {}
    with get_conn() as conn:
        indexed = {
            r["uuid"]: (r["source_sha256"], r["updated_at"])
            for r in conn.execute(
                "SELECT uuid, source_sha256, updated_at FROM conversations"
            ).fetchall()
        }

        progress = tqdm(convs, desc="Importing", file=sys.stderr, disable=not verbose)
        for i, conv in enumerate(progress):
            uuid = conv["uuid"]
            messages = conv.get("chat_messages") or []
            digest = parse.messages_sha256(messages)
            updated = _parse_ts(conv.get("updated_at"))

            # An older export must never overwrite a newer one. Exports overlap, so
            # importing a full export after an incremental would otherwise roll
            # conversations back to their stale snapshots and silently drop messages.
            if uuid in indexed:
                known_updated = indexed[uuid][1]
                if updated and known_updated and updated < known_updated:
                    statuses[uuid] = STALE
                    continue

            # Metadata is cheap, so refresh it every time: a rename or a re-summarized
            # conversation lands without touching the transcript.
            conn.execute(
                _UPSERT_CONV_META,
                {
                    "uuid": uuid,
                    "name": conv.get("name") or None,
                    "summary": conv.get("summary") or None,
                    "created_at": _parse_ts(conv.get("created_at")),
                    "updated_at": updated,
                    "account_uuid": (conv.get("account") or {}).get("uuid"),
                },
            )

            if uuid not in indexed:
                status = NEW
            elif indexed[uuid][0] == digest and not reimport:
                statuses[uuid] = UNCHANGED
                continue
            else:
                known = {
                    r["uuid"]
                    for r in conn.execute(
                        "SELECT uuid FROM messages WHERE conv_uuid = %(uuid)s",
                        {"uuid": uuid},
                    ).fetchall()
                }
                status = GREW if {m.get("uuid") for m in messages} - known else CHANGED
            statuses[uuid] = status

            # Everything below re-derives this conversation from scratch.
            conn.execute(
                "DELETE FROM messages WHERE conv_uuid = %(uuid)s",
                {"uuid": uuid},
            )
            upload_views = parse.view_upload_names(messages)
            rows, chunk_rows = [], []
            for seq, msg in enumerate(messages):
                # Every message gets a row, even one with nothing indexable in it: the
                # conversation is a tree, and a text-less message can still be an
                # interior node whose children would otherwise dangle.
                prose, tool = parse.message_texts(msg, upload_views)
                muuid = msg.get("uuid") or f"{uuid}:{seq}"
                created = _parse_ts(msg.get("created_at"))
                sender = msg.get("sender")
                rows.append({
                    "uuid": muuid,
                    "conv_uuid": uuid,
                    "seq": seq,
                    "parent_uuid": parse.parent_uuid(msg),
                    "sender": sender,
                    "created_at": created,
                    "text": prose,
                    "tool_text": tool or None,
                    "raw": Jsonb(msg),
                })
                # Only prose is embedded. A short message yields one chunk; a long
                # pasted document yields several (so semantic search covers it all);
                # prose too short to mean anything yields none, so no vector is ever
                # computed for it and it stays out of the category centroids.
                if len(prose.strip()) >= config.EMBEDDING_MIN_CHARS:
                    for cseq, chunk in enumerate(parse.chunk_text(prose)):
                        chunk_rows.append({
                            "msg_uuid": muuid,
                            "conv_uuid": uuid,
                            "seq": cseq,
                            "sender": sender,
                            "created_at": created,
                            "text": chunk,
                            "text_sha256": parse.text_sha256(chunk),
                        })
            if rows:
                conn.cursor().executemany(_INSERT_MSG, rows)
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
            if (i + 1) % 200 == 0:
                conn.commit()

        # The rest of the zip: small, unparsed, stored because it is already here.
        artifacts = read_artifacts(path)
        for kind, art_uuid, obj in artifacts:
            conn.execute(
                _UPSERT_ARTIFACT,
                {
                    "kind": kind,
                    "uuid": art_uuid,
                    "raw": Jsonb(obj),
                },
            )
        if verbose and artifacts:
            kinds = sorted({k for k, _, _ in artifacts})
            print(f"Stored {len(artifacts)} export artifacts ({', '.join(kinds)})", file=sys.stderr)

        # Re-apply category tags from the curation file (rebuildable index).
        # Tags are auxiliary -- never fail an import over them.
        try:
            categories.sync_to_db(conn)
        except Exception as exc:
            print(f"Warning: category sync skipped ({exc})", file=sys.stderr)

    return {
        "conversations": len(convs),
        "messages": n_msgs,
        "statuses": statuses,
    }
