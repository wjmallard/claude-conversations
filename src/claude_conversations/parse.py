"""Read and interpret the exported conversation files.

Export layout (one pair per conversation, in CONVERSATIONS_DIR):
  <uuid>.metadata.json  -> {uuid, name, summary, created_at, updated_at, account:{uuid}}
  <uuid>.jsonl          -> one message object per line, in conversation order:
       {uuid, sender, created_at, parent_message_uuid, text, content:[...blocks...],
        attachments:[...], files:[...], ...}

A message's `content` is a list of typed blocks. The flattened top-level `text`
field is unreliable for display (it contains "block not supported" placeholders
where tools ran), so we always work from `content`. Block types seen in the data:
  text, thinking, tool_use (name+input+message), tool_result (name+content),
  token_budget (skip), flag (rare). Images are ignored by design.

Pasted/uploaded documents are NOT content blocks -- they live in the message-level
`attachments` list ({file_name, file_type, file_size, extracted_content}); the
`files` list holds content-free references to binary uploads (images). See
message_texts() for how attachment text is folded into the searchable streams.
"""

import hashlib
import json
import os
import re
from pathlib import Path

META_SUFFIX = ".metadata.json"
JSONL_SUFFIX = ".jsonl"

# Attachment routing: prose documents (txt/md/docx/pdf/...) join the embedded prose
# stream; everything else (pasted source code, TSV/CSV, JSON/YAML, logs) is routed
# to the non-embedded tool text -- searchable, but kept out of the semantic centroids.
_PROSE_ATTACHMENT_TYPES = {
    "application/pdf",
    "doc",
    "docx",
    "html",
    "md",
    "pdf",
    "rtf",
    "text/html",
    "text/markdown",
    "text/plain",
    "txt",
}
_PROSE_ATTACHMENT_SUFFIXES = (
    ".doc",
    ".docx",
    ".htm",
    ".html",
    ".md",
    ".pdf",
    ".rtf",
    ".txt",
)

# Max characters per embedding chunk. The MLX embedder truncates at 8192 tokens
# (~3-4 chars/token); 24k chars stays comfortably under that, so a long pasted
# document is split into fully-embedded chunks instead of being silently cut off.
EMBED_CHUNK_CHARS = 24000

# Newer claude.ai uploads land in /mnt/user-data/uploads/ and are read via the `view`
# tool; that content (the user's own documents) is promoted from tool text into the
# embedded prose stream -- see view_upload_names / message_texts.
_UPLOADS_PREFIX = "/mnt/user-data/uploads/"
_LINENO_RE = re.compile(r"(?m)^ *\d+\t")


def iter_conversation_files(conversations_dir):
    """Yield (uuid, jsonl_path, meta_path) for every conversation in the dir.

    Driven by the .metadata.json files; a missing .jsonl yields path-not-exists
    and is handled by the caller (still indexed as an empty conversation).
    """
    d = Path(conversations_dir)
    for meta_path in sorted(d.glob("*" + META_SUFFIX)):
        uuid = meta_path.name[: -len(META_SUFFIX)]
        jsonl_path = d / (uuid + JSONL_SUFFIX)
        yield uuid, jsonl_path, meta_path


def load_metadata(meta_path) -> dict:
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_messages(jsonl_path) -> list[dict]:
    """Parse a .jsonl transcript into a list of message dicts, in file order."""
    out = []
    try:
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return out


def _tool_result_text(block) -> str:
    """Pull plain text out of a tool_result block's content (web results, text, etc.)."""
    content = block.get("content")
    parts = []
    items = content if isinstance(content, list) else [content]
    for it in items:
        if isinstance(it, dict):
            if it.get("title"):
                parts.append(it["title"])
            if it.get("text"):
                parts.append(it["text"])
            url = (it.get("metadata") or {}).get("site_name") or it.get("url")
            if url:
                parts.append(url)
        elif isinstance(it, str):
            parts.append(it)
    return "\n".join(p for p in parts if p)


def _attachment_is_prose(file_type, file_name) -> bool:
    """Whether an attachment is a prose document (vs. code/data). Prose feeds the
    embedded prose stream; everything else is routed to the non-embedded tool text."""
    if (file_type or "").lower() in _PROSE_ATTACHMENT_TYPES:
        return True
    return (file_name or "").lower().endswith(_PROSE_ATTACHMENT_SUFFIXES)


def _strip_line_numbers(s: str) -> str:
    """Strip the `cat -n`-style 'NNN\\t' prefixes the view tool prepends to file lines."""
    return _LINENO_RE.sub("", s or "")


def view_upload_names(messages) -> dict:
    """Map tool_use_id -> filename for `view` tool calls that read an UPLOADED file
    (path under /mnt/user-data/uploads/). Conversation-level, so message_texts can route
    those file-view tool_results into embedded prose instead of (unembedded) tool text."""
    out = {}
    for m in messages:
        for b in m.get("content") or []:
            if not isinstance(b, dict) or b.get("type") != "tool_use" or b.get("name") != "view":
                continue
            inp = b.get("input") or {}
            path = inp.get("path") or inp.get("file_path") or ""
            if _UPLOADS_PREFIX in path and b.get("id"):
                out[b["id"]] = os.path.basename(path.rstrip("/")) or "uploaded-file"
    return out


def message_texts(msg, upload_views=None) -> tuple[str, str]:
    """Split a message's content blocks by provenance: returns (prose, tool).

    prose -- human-typed + assistant prose (text blocks), plus the extracted text of
            any pasted *prose* documents. This is the real conversation: it drives
            full-text/fuzzy/semantic search, categorization, and embeddings.
            Combined with the `sender` column it also isolates what the *user*
            actually contributed (sender=human, non-empty prose) from tool-results
            that arrive as human-role messages.
    tool  -- tool_use (name + JSON input) + tool_result text, plus the text of pasted
            *code/data* documents. Stored separately so it stays searchable, but it
            is never embedded (machine/code text isn't "what a conversation is about").

    Pasted/uploaded documents live in msg["attachments"] (not content blocks); each
    is folded in with an inline "[attachment: name]" label and routed by file type
    (see _attachment_is_prose). Content-free uploads (msg["files"], images) are
    ignored here and only surface as filenames in the detail view.

    Thinking, token_budget, flag, and images are dropped -- not indexed (they remain
    on disk and are rendered in the detail view). Either string may be ''.
    """
    prose, tool = [], []
    for b in msg.get("content") or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            if b.get("text"):
                prose.append(b["text"])
        elif t == "tool_use":
            if b.get("name"):
                tool.append(b["name"])
            inp = b.get("input")
            if inp:
                tool.append(json.dumps(inp, ensure_ascii=False))
        elif t == "tool_result":
            txt = _tool_result_text(b)
            if txt:
                tuid = b.get("tool_use_id")
                if upload_views and tuid in upload_views and not b.get("is_error"):
                    prose.append(f"[uploaded file: {upload_views[tuid]}]\n{_strip_line_numbers(txt)}")
                else:
                    tool.append(txt)
        # thinking / token_budget / flag / image / unknown -> not indexed

    # Fold in pasted/uploaded documents, routed by type and labeled so they read as
    # attachments in search results.
    for a in msg.get("attachments") or []:
        if not isinstance(a, dict):
            continue
        content = (a.get("extracted_content") or "").strip()
        if not content:
            continue
        block = f"[attachment: {a.get('file_name') or 'document'}]\n{content}"
        if _attachment_is_prose(a.get("file_type"), a.get("file_name")):
            prose.append(block)
        else:
            tool.append(block)

    prose_text = "\n".join(prose).strip()
    tool_text = "\n".join(tool).strip()
    # Fallback: a message with no content blocks at all -> use the flattened
    # top-level text as prose (best effort; carries placeholders where tools ran).
    if not prose_text and not tool_text and msg.get("text"):
        prose_text = str(msg["text"]).strip()
    return prose_text, tool_text


def chunk_text(text, max_chars=EMBED_CHUNK_CHARS) -> list[str]:
    """Split prose into chunks of at most max_chars, so long messages (pasted
    documents) are embedded in full rather than truncated. Splits on a paragraph
    break, then a line break, then a space near the cap; falls back to a hard cut.
    Short text returns a single chunk; empty text returns no chunks."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks, start, n = [], 0, len(text)
    while start < n:
        if n - start <= max_chars:
            chunks.append(text[start:])
            break
        window = text[start : start + max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 2:
            cut = window.rfind("\n")
        if cut < max_chars // 2:
            cut = window.rfind(" ")
        if cut < max_chars // 2:
            cut = max_chars  # no usable boundary; hard cut
        chunks.append(text[start : start + cut])
        start += cut
    return [c for c in (c.strip() for c in chunks) if c]


def file_sha256(path) -> str:
    """Return the hex SHA-256 of a file's bytes for incremental-reindex comparison,
    or '' if it does not exist.

    Content, not mtime: a fresh export rewrites every file with a new timestamp, so
    mtime+size bookkeeping reports the whole archive as changed and forces a full
    rebuild -- which drops every embedding through the message_chunks cascade. Only a
    genuinely changed transcript changes its digest, and digesting the entire archive
    costs a couple of seconds.
    """
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except FileNotFoundError:
        return ""
