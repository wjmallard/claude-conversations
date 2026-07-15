-- claude-conversations schema.
--
-- The filesystem (exported *.jsonl + *.metadata.json) is the source of truth.
-- This database is a rebuildable index over it: typed columns + full-text
-- (tsvector), fuzzy (pg_trgm), and semantic (pgvector) search. Raw message
-- content is NOT stored here -- the detail view re-reads the .jsonl from disk
-- and renders it. We persist only the plain text needed to drive search.

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
    -- incremental-reindex bookkeeping: skip transcripts whose CONTENT is unchanged.
    -- Content, not mtime: a fresh export rewrites every file with a new timestamp,
    -- so mtime+size would rebuild the whole archive and drop every embedding via
    -- the message_chunks cascade.
    source_path   TEXT,
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

-- mtime+size bookkeeping replaced by the content digest (see source_sha256 above).
-- Digest-first: back-fill source_sha256 BEFORE applying this, or the first cc-index
-- rebuilds every conversation and re-embeds the whole archive.
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS source_sha256 TEXT;
ALTER TABLE conversations DROP COLUMN IF EXISTS source_mtime;
ALTER TABLE conversations DROP COLUMN IF EXISTS source_size;

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

-- One embedding per chunk of a message's PROSE. Long pasted documents are split
-- into <=24k-char chunks so semantic search covers them in full (the MLX embedder
-- truncates at ~8k tokens); a short message produces exactly one chunk. conv_uuid,
-- sender, and created_at are denormalized from the parent message so semantic search
-- and the category centroids apply the same filters as keyword search.
CREATE TABLE IF NOT EXISTS message_chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    msg_uuid    TEXT NOT NULL REFERENCES messages(uuid) ON DELETE CASCADE,
    conv_uuid   TEXT NOT NULL REFERENCES conversations(uuid) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,            -- chunk index within the message
    sender      TEXT,                        -- copied from the parent message
    created_at  TIMESTAMPTZ,                 -- copied from the parent message
    text        TEXT NOT NULL,               -- a <=24k-char slice of messages.text (prose)
    embedding   vector(1024)                 -- MLX Qwen3-Embedding-0.6B; NULL until `cc-embed`
);

CREATE INDEX IF NOT EXISTS idx_chunk_conv      ON message_chunks(conv_uuid);
CREATE INDEX IF NOT EXISTS idx_chunk_msg       ON message_chunks(msg_uuid);
CREATE INDEX IF NOT EXISTS idx_chunk_embedding ON message_chunks USING hnsw (embedding vector_cosine_ops);
