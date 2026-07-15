-- claude-conversations schema.
--
-- The claude.ai export .zip is the source of truth. This database is a rebuildable
-- index over it: typed columns + full-text (tsvector), fuzzy (pg_trgm), and semantic
-- (pgvector) search, plus every message exactly as exported (messages.raw). cc-import
-- reads a zip and writes all of this; nothing here is ever read back from disk, and
-- there is no intermediate copy of the archive to drift out of sync with the export.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS conversations (
    uuid          TEXT PRIMARY KEY,
    name          TEXT,
    summary       TEXT,
    created_at    TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ,
    account_uuid  TEXT,
    n_messages    INTEGER NOT NULL DEFAULT 0,
    -- Digest of this conversation's messages AS EXPORTED. Re-importing an export
    -- already indexed is a no-op: the digest matches, the transcript is not rebuilt,
    -- and its cached embeddings are never disturbed.
    source_sha256 TEXT,
    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- multi-label category tags; rebuildable mirror of categories.json
    categories    TEXT[] NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_conv_created   ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conv_updated   ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conv_name_trgm ON conversations USING GIN (name gin_trgm_ops);

-- Idempotent migration for pre-existing databases, plus the tag-filter index.
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS categories TEXT[] NOT NULL DEFAULT '{}';
CREATE INDEX IF NOT EXISTS idx_conv_categories ON conversations USING GIN (categories);

-- mtime+size bookkeeping replaced by a content digest; source_path is meaningless now
-- that the export is read straight from the zip and never staged on disk.
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS source_sha256 TEXT;
ALTER TABLE conversations DROP COLUMN IF EXISTS source_mtime;
ALTER TABLE conversations DROP COLUMN IF EXISTS source_size;
ALTER TABLE conversations DROP COLUMN IF EXISTS source_path;

-- Everything else the export ships: users.json, memories.json, projects/<uuid>.json,
-- reflections/<uuid>.json. Kept verbatim and unparsed -- nothing reads them yet, but
-- they are in the file, so leaving them on the floor would just mean re-importing
-- later to get them back. `uuid` is '' for the singletons (users, memories).
CREATE TABLE IF NOT EXISTS export_artifacts (
    kind        TEXT NOT NULL,          -- 'users' | 'memories' | 'projects' | ...
    uuid        TEXT NOT NULL,          -- '' when the export ships exactly one
    raw         JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, uuid)
);

-- One row per message -- EVERY message, including those carrying no extractable text.
-- A conversation is a tree (see parent_uuid), and a message with nothing indexable can
-- still be an interior node, so skipping it would break the chain below it.
-- Search operates on messages (filtering text <> ''), then aggregates the best score
-- per conversation.
CREATE TABLE IF NOT EXISTS messages (
    uuid        TEXT PRIMARY KEY,
    conv_uuid   TEXT NOT NULL REFERENCES conversations(uuid) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,            -- position within the conversation (file order)
    -- Parent in the conversation tree; NULL marks a conversation HEAD. There may be
    -- SEVERAL heads: revising an opening prompt forks at the root. The export's root
    -- sentinel is normalized away -- see parse.parent_uuid.
    parent_uuid TEXT,
    sender      TEXT,                        -- 'human' | 'assistant'
    created_at  TIMESTAMPTZ,
    text        TEXT NOT NULL,               -- PROSE only: human-typed + assistant text; '' when none
    text_tsv    TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    tool_text   TEXT,                        -- tool_use inputs + tool_result text; not embedded
    -- The message exactly as exported. The UI renders from this, so browsing no longer
    -- re-reads the .jsonl (and works with the archive drive unmounted). The filesystem
    -- is still the source of truth: cc-index rebuilds this column from it.
    raw         JSONB
);

-- Idempotent migration for pre-existing databases.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool_text TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS parent_uuid TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS raw JSONB;
-- Embeddings moved to message_chunks (long pasted documents are split into chunks);
-- drop the old per-message embedding column + its index when upgrading.
ALTER TABLE messages DROP COLUMN IF EXISTS embedding;

CREATE INDEX IF NOT EXISTS idx_msg_conv      ON messages(conv_uuid, seq);
CREATE INDEX IF NOT EXISTS idx_msg_created   ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_msg_parent    ON messages(parent_uuid);
CREATE INDEX IF NOT EXISTS idx_msg_text_tsv  ON messages USING GIN (text_tsv);
CREATE INDEX IF NOT EXISTS idx_msg_text_trgm ON messages USING GIN (text gin_trgm_ops);

-- One row per chunk of a message's PROSE. Long pasted documents are split into
-- <=24k-char chunks so semantic search covers them in full (the MLX embedder truncates
-- at ~8k tokens, and the archive holds messages many times that); a short message
-- produces exactly one chunk, and prose too short to carry retrievable meaning
-- (embedding_min_chars) produces none. conv_uuid, sender, and created_at are
-- denormalized from the parent message so semantic search and the category centroids
-- apply the same filters as keyword search.
CREATE TABLE IF NOT EXISTS message_chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    msg_uuid    TEXT NOT NULL REFERENCES messages(uuid) ON DELETE CASCADE,
    conv_uuid   TEXT NOT NULL REFERENCES conversations(uuid) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,            -- chunk index within the message
    sender      TEXT,                        -- copied from the parent message
    created_at  TIMESTAMPTZ,                 -- copied from the parent message
    text        TEXT NOT NULL,               -- a <=24k-char slice of messages.text (prose)
    text_sha256 TEXT NOT NULL                -- joins to embeddings; see below
);

-- Idempotent migration for pre-existing databases: embeddings moved out of
-- message_chunks and are now keyed by content, so a rebuild no longer discards them.
-- Must precede the indexes below, which reference the new column.
ALTER TABLE message_chunks ADD COLUMN IF NOT EXISTS text_sha256 TEXT;
ALTER TABLE message_chunks DROP COLUMN IF EXISTS embedding;
DROP INDEX IF EXISTS idx_chunk_embedding;

CREATE INDEX IF NOT EXISTS idx_chunk_conv ON message_chunks(conv_uuid);
CREATE INDEX IF NOT EXISTS idx_chunk_msg  ON message_chunks(msg_uuid);
CREATE INDEX IF NOT EXISTS idx_chunk_sha  ON message_chunks(text_sha256);

-- The vector for one piece of prose, keyed by the prose itself.
--
-- An embedding is a pure function of its text, so it is cached by content rather than
-- owned by a chunk row. Re-indexing a conversation deletes and recreates its chunks (a
-- cascade from messages), which would otherwise throw away every vector and force a
-- re-embed -- even when an export merely re-serialized a transcript and not a word of
-- the prose changed. Nothing cascades into this table, so the vectors outlive rebuilds
-- and cc-embed only computes text it has never seen.
--
-- No HNSW index here, deliberately. Semantic search is a THRESHOLD query (every chunk
-- above a similarity floor, aggregated per conversation), and pgvector can only serve
-- HNSW for `ORDER BY embedding <=> q LIMIT k`. The old index cost 427 MB and the
-- planner never used it. Add one only alongside a query that can.
CREATE TABLE IF NOT EXISTS embeddings (
    text_sha256 TEXT PRIMARY KEY,           -- sha256 of message_chunks.text
    embedding   vector(1024) NOT NULL       -- MLX Qwen3-Embedding-0.6B
);
