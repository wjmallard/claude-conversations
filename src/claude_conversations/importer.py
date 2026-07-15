"""Split a claude.ai data export into the filesystem archive.

An export is a .zip holding one conversations.json -- a JSON array of conversation
objects, each carrying its messages in `chat_messages` -- alongside users/projects/
memories files this tool ignores. It reads straight out of the zip in about a second,
so there is no extraction step and nothing external to install.

Each conversation becomes a self-contained pair, which cc-index then reads:

  <uuid>.jsonl          one message per line, verbatim (every field preserved)
  <uuid>.metadata.json  the conversation minus chat_messages, pretty-printed

MERGE, never truncate. claude.ai issues incremental exports covering only a recent
window, so an export is usually a slice of the archive rather than a snapshot of it.
Conversations an export does not mention are left untouched: importing an incremental
adds and updates, and never deletes history.

Serialization is compact with raw UTF-8, which reproduces the earlier jq-based
splitter byte for byte. Re-importing an export already in the archive therefore
rewrites nothing, and cc-index skips every conversation.
"""

import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

from claude_conversations import config, parse

_CONVERSATIONS_MEMBER = "conversations.json"

# write_conversation() outcomes.
NEW = "new"              # not in the archive yet
GREW = "grew"            # present, and the export carries messages the archive lacks
CHANGED = "changed"      # present with the same messages, but their content differs
UNCHANGED = "unchanged"  # byte-for-byte what the archive already holds


def read_export(path) -> list[dict]:
    """Return the conversation list from an export .zip, or from a bare conversations.json."""
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


def _jsonl_bytes(messages) -> bytes:
    """Serialize messages as JSONL: one compact, raw-UTF-8 object per line.

    Message text can carry a raw U+2028 LINE SEPARATOR, which is legal inside a JSON
    string and does not end the line. Readers must iterate the file -- which breaks on
    \\n alone -- and never str.splitlines(), which splits on U+2028 too and would tear
    an object in half.
    """
    return b"".join(orjson.dumps(m) + b"\n" for m in messages)


def _write_atomic(path, data: bytes):
    """Write via temp file + rename, so an interrupted import never leaves a
    half-written conversation in the archive."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_conversation(outdir, conv) -> str:
    """Write one conversation's pair into outdir; returns NEW/GREW/CHANGED/UNCHANGED.

    A conversation with no messages still gets an (empty) .jsonl, so the indexer sees
    it rather than silently dropping it.
    """
    uuid = conv["uuid"]
    jsonl = Path(outdir) / (uuid + parse.JSONL_SUFFIX)
    meta_path = Path(outdir) / (uuid + parse.META_SUFFIX)

    messages = conv.get("chat_messages") or []
    new_bytes = _jsonl_bytes(messages)
    old_bytes = jsonl.read_bytes() if jsonl.exists() else None

    if old_bytes is None:
        status = NEW
    elif old_bytes == new_bytes:
        status = UNCHANGED
    else:
        # Separate a conversation that CONTINUED from one the export merely
        # re-serialized: claude.ai has added fields to existing blocks before, which
        # moves the bytes without changing a word anyone wrote.
        known = {m.get("uuid") for m in parse.load_messages(jsonl)}
        status = GREW if {m.get("uuid") for m in messages} - known else CHANGED

    if status != UNCHANGED:
        _write_atomic(jsonl, new_bytes)

    # The sidecar is tiny and moves on its own -- renaming a conversation leaves its
    # transcript untouched -- so refresh it whenever it differs.
    meta_bytes = orjson.dumps(
        {k: v for k, v in conv.items() if k != "chat_messages"},
        option=orjson.OPT_INDENT_2,
    ) + b"\n"
    if not meta_path.exists() or meta_path.read_bytes() != meta_bytes:
        _write_atomic(meta_path, meta_bytes)

    return status


def import_export(path, outdir=None, verbose=True) -> dict:
    """Split an export into outdir (default config.CONVERSATIONS_DIR). Returns counts."""
    outdir = Path(outdir or config.CONVERSATIONS_DIR).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    convs = read_export(path)
    if verbose:
        n_msgs = sum(len(c.get("chat_messages") or []) for c in convs)
        print(
            f"Export holds {len(convs)} conversations, {n_msgs} messages",
            file=sys.stderr,
        )

    counts = Counter()
    progress = tqdm(convs, desc="Splitting", file=sys.stderr, disable=not verbose)
    for conv in progress:
        counts[write_conversation(outdir, conv)] += 1

    # Whatever the export did not mention stays as it is: an incremental covers only a
    # window, and the rest of the archive is still the archive.
    archive_total = sum(1 for _ in outdir.glob("*" + parse.META_SUFFIX))
    stats = {
        "archive_total": archive_total,
        "changed": counts[CHANGED],
        "conversations": len(convs),
        "grew": counts[GREW],
        "new": counts[NEW],
        "unchanged": counts[UNCHANGED],
        "untouched": archive_total - len(convs),
    }
    if verbose:
        print(
            f"  {stats['new']:6d} new\n"
            f"  {stats['grew']:6d} gained messages\n"
            f"  {stats['changed']:6d} changed with no new messages\n"
            f"  {stats['unchanged']:6d} unchanged\n"
            f"  {stats['untouched']:6d} not in this export (left alone)\n"
            f"Archive now holds {archive_total} conversations.",
            file=sys.stderr,
        )
    return stats
