# claude-conversations

Local browser + search over exported Claude conversations. Sibling of, and
modeled on, the `twitter-news` project (same stack and conventions).

## Architecture

- Python 3, Flask, PostgreSQL (psycopg3, local peer auth), pgvector + pg_trgm.
- **The filesystem is the source of truth; the database is a rebuildable index.**
  Raw message content is NOT stored in the DB — the detail view re-reads the
  `.jsonl` from disk and renders it. The DB stores typed metadata and per-message
  text split by **provenance**: `text` = prose (human-typed + assistant text),
  which drives search + embeddings; `tool_text` = tool_use/tool_result I/O, stored
  but never embedded. Thinking is not indexed. Embeddings cover prose only.
- Semantic search uses a **local MLX embedding model** (Qwen3-Embedding-0.6B,
  `vector(1024)`, cosine via HNSW) — no API cost. MLX deps are an optional
  `semantic` extra so the base tool stays light.
- **Out of scope:** OCR, translation, images/media. Do not add them.

## Data model

Export = one pair per conversation in `conversations_dir`:
- `<uuid>.metadata.json` → `{uuid, name, summary, created_at, updated_at, account:{uuid}}`
- `<uuid>.jsonl` → one message per line: `{uuid, sender, created_at, parent_message_uuid, text, content:[blocks]}`

Always render from `content` blocks, never the flattened top-level `text` (it
contains "block not supported" placeholders where tools ran). Block types:
`text`, `thinking`, `tool_use`, `tool_result`, `token_budget` (skip), `flag`.

Pasted/uploaded documents are NOT content blocks — they live in the message-level
`attachments` list and are routed by file type: prose documents join the embedded
prose stream, code/data join `tool_text`. The `files` list holds content-free
references to binary uploads (images) and is ignored.

Tables: `conversations` (metadata + bookkeeping + `categories` tags), `messages`
(one row per message: `text` = prose, generated `text_tsv`, `tool_text` = tool
I/O), and `message_chunks` (prose split into ≤24k-char slices, one `embedding`
per chunk — the MLX embedder truncates at ~8k tokens, so a long pasted document
is chunked rather than silently cut off). Keyword and fuzzy search query
`messages`; semantic queries `message_chunks`. All three aggregate the best score
per conversation. `sender='human' AND text<>''` isolates what the user actually
typed.

## Layout

- `src/claude_conversations/`: `config.py`, `db.py` (connection + search), `parse.py`
  (read/flatten export), `indexer.py`, `embedding.py` (MLX, lazy import),
  `render.py` (content blocks → HTML), `categories.py` (tag curation store),
  `classifier.py` (layered keyword/semantic sieve), `cli.py`, `web/` (Flask app +
  templates).
- `sql/schema.sql`: idempotent (`IF NOT EXISTS`); `cc-initdb` applies it.
- Config in `config.yaml` (see `config.yaml.example`); tag curation in
  `categories.json` — on-disk source of truth, mirrored to `conversations.categories`.

## Conventions

- psycopg uses `%`-style params, so the pg_trgm `<<%` operator is written `<<%%`.
- Snippets highlight via sentinels (`\x02`/`\x03`) so HTML is escaped in the web
  layer before `<mark>` is inserted — never inject markup from SQL.
- `cc-index` and `cc-embed` are incremental/resumable; safe to re-run.

## Formatting (clean, reviewable diffs)

- **One item per line** in any multi-line construct, with a trailing comma:
  - function calls/defs with many args — one argument per line, closing `)` on its own line
  - dicts and lists — one key/element per line
  - multi-line SQL — triple-quoted; one clause per line (`SELECT`/`FROM`/`JOIN`/`WHERE`/`GROUP BY`/`ORDER BY`/`LIMIT`); one column per line in `SELECT` (`col AS alias` stays together); extra `WHERE` conditions begin their own line with `AND`/`OR`; the param dict has one key per line
  - long imports — parenthesized, one name per line
- Preserve alphabetization in already-alphabetized structures (dicts, lists, maps).
- SQL: named (dict-style) params, not positional; alias ambiguous columns; a bare row count is `SELECT count(*)` read as `row["count"]` (alias other aggregates descriptively, e.g. `count(*) AS tweet_count`); access rows via `row["key"]`. Canonical style: twitter-news `src/twitter_news/db.py`.
- No linter in use — never add `# noqa` / `# type: ignore`.
