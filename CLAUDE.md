# claude-conversations

Local browser + search over exported Claude conversations. Sibling of, and
modeled on, the `twitter-news` project (same stack and conventions).

## Architecture

- Python 3, Flask, PostgreSQL (psycopg3, local peer auth), pgvector + pg_trgm.
- **The export .zip is the source of truth; the database is a rebuildable index.**
  `cc-import` reads a zip and everything lands in Postgres -- there is NO archive on
  disk and nothing is ever read back from one. (There used to be: a directory of split
  `.jsonl` files. It held nothing the DB lacks, and its only lasting effect was drifting
  out of sync with the export it came from. Do not reintroduce it; if plain files are
  ever wanted, generate them on demand.)
- `messages.raw` holds each message exactly as exported, so the UI renders from the DB.
  Alongside it the DB stores typed metadata and per-message text split by
  **provenance**: `text` = prose (human-typed + assistant text), which drives search +
  embeddings; `tool_text` = tool_use/tool_result I/O, stored but never embedded.
  Thinking is not indexed (but IS preserved in `raw`). Embeddings cover prose only.
- Semantic search uses a **local MLX embedding model** (Qwen3-Embedding-0.6B,
  `vector(1024)`, cosine) -- no API cost. MLX deps are an optional `semantic` extra so
  the base tool stays light. There is deliberately no HNSW index: search is a threshold
  query, which pgvector can only serve from a seq scan.
- **Out of scope:** OCR, translation, images/media. Do not add them.

## Data model

A claude.ai export is a `.zip` holding one big `conversations.json` (a JSON array of
conversations, each with its messages inline under `chat_messages`), plus small side
files (users, memories, projects, reflections) kept verbatim in `export_artifacts`.

Exports are often **incremental** (a recent window, not a snapshot), so importing
merges: a conversation an export does not mention is left alone, never deleted -- an
incremental cannot tell "deleted upstream" from "outside the window". Import order is
irrelevant: a conversation whose export `updated_at` predates what is indexed is
ignored, so an old export cannot roll a newer one back.

Always render from `content` blocks, never the flattened top-level `text` (it
contains "block not supported" placeholders where tools ran). Block types:
`text`, `thinking`, `tool_use`, `tool_result`, `token_budget` (skip), `flag`.

Pasted/uploaded documents are NOT content blocks -- they live in the message-level
`attachments` list and are routed by file type: prose documents join the embedded
prose stream, code/data join `tool_text`. The `files` list holds content-free
references to binary uploads (images) and is ignored.

Tables: `conversations` (metadata + bookkeeping + `categories` tags), `messages`
(one row per message: `text` = prose, generated `text_tsv`, `tool_text` = tool
I/O), and `message_chunks` (prose split into <=24k-char slices, one `embedding`
per chunk -- the MLX embedder truncates at ~8k tokens, so a long pasted document
is chunked rather than silently cut off). Keyword and fuzzy search query
`messages`; semantic queries `message_chunks`. All three aggregate the best score
per conversation. `sender='human' AND text<>''` isolates what the user actually
typed.

## Layout

- `src/claude_conversations/`: `config.py`, `db.py` (connection + search), `importer.py`
  (export .zip -> Postgres, via orjson + stdlib zipfile; the whole ingest path),
  `parse.py` (interpret conversation/message objects; no file I/O), `embedding.py`
  (MLX, lazy import), `render.py` (content blocks -> HTML), `categories.py` (tag
  curation store), `classifier.py` (layered keyword/semantic sieve), `cli.py`,
  `web/` (Flask app + templates).
- `sql/schema.sql`: idempotent (`IF NOT EXISTS`); `cc-initdb` applies it.
- Config in `config.yaml` (see `config.yaml.example`); tag curation in
  `categories.json` -- on-disk source of truth, mirrored to `conversations.categories`.

## Conventions

- psycopg uses `%`-style params, so the pg_trgm `<<%` operator is written `<<%%`.
- Snippets highlight via sentinels (`\x02`/`\x03`) so HTML is escaped in the web
  layer before `<mark>` is inserted -- never inject markup from SQL.
- `cc-import` and `cc-embed` are incremental/resumable; safe to re-run. Import is split
  by cost: conversation metadata refreshes every run (a rename lands for free), while
  rebuilding a transcript's messages + chunks happens only when
  `conversations.source_sha256` changes. Detect changes by **content, never a
  timestamp** -- an export re-serializes everything, and upstream has silently added
  fields to existing blocks before.
- Rebuilding a conversation deletes and recreates its chunks (a cascade from
  `messages`), so an embedding must never be owned by a chunk row. Vectors live in
  `embeddings`, keyed by `sha256(text)`, which nothing cascades into -- an embedding is
  a pure function of its text, so a rebuild costs no re-embedding.
- Report against the **database**, not against intermediate state. Users care what
  landed in the index they search.

## Formatting (clean, reviewable diffs)

- **One item per line** in any multi-line construct, with a trailing comma:
  - function calls/defs with many args -- one argument per line, closing `)` on its own line
  - dicts and lists -- one key/element per line
  - multi-line SQL -- triple-quoted; one clause per line (`SELECT`/`FROM`/`JOIN`/`WHERE`/`GROUP BY`/`ORDER BY`/`LIMIT`); one column per line in `SELECT` (`col AS alias` stays together); extra `WHERE` conditions begin their own line with `AND`/`OR`; the param dict has one key per line
  - long imports -- parenthesized, one name per line
- Preserve alphabetization in already-alphabetized structures (dicts, lists, maps).
- SQL: named (dict-style) params, not positional; alias ambiguous columns; a bare row count is `SELECT count(*)` read as `row["count"]` (alias other aggregates descriptively, e.g. `count(*) AS tweet_count`); access rows via `row["key"]`. Canonical style: twitter-news `src/twitter_news/db.py`.
- No linter in use -- never add `# noqa` / `# type: ignore`.
