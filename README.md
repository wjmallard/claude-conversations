# claude-conversations

An offline browser and search engine for exported Claude conversations.

- **Browse** conversations, sorted by date / title / length.
- **Search** with three modes:
  - **keyword** -- Postgres full-text (`tsvector`), stemmed, with highlighted snippets
  - **fuzzy** -- `pg_trgm` trigram similarity, typo- and substring-tolerant
  - **semantic** -- local MLX embeddings (Qwen3) + `pgvector` cosine, concept search
- **Read** a conversation with Markdown rendering and collapsible thinking / tool-call / tool-result blocks.

## Setup

Requires PostgreSQL with the `pg_trgm` and `vector` extensions, and (for semantic
search) Apple Silicon for the local MLX embedder.

Get your archive from claude.ai: **Settings -> Privacy -> Export data**. You will be
emailed a `.zip`; point `cc-import` at it.

```sh
cp config.yaml.example config.yaml   # then edit db_name if you like
uv sync                              # base deps (keyword + fuzzy + browse)
uv run cc-initdb                     # create database + schema
uv run cc-import ~/Downloads/data-*.zip   # read the zip into Postgres
uv run cc-web                        # serve at http://127.0.0.1:5005
```

`cc-import` reads `conversations.json` out of the zip and writes everything to the
database -- typed metadata, every message exactly as exported, and the split text
that drives search -- plus the export's other files (users, memories, projects,
reflections) stored verbatim.

### Semantic search (optional)

```sh
uv sync --extra semantic             # mlx + mlx-lm
uv run cc-embed                      # embed messages locally
```

Then select **semantic** mode in the search bar.

## Commands

| Command | What it does |
| --- | --- |
| `cc-initdb` | Create the database and apply `sql/schema.sql`; idempotent, safe to re-run |
| `cc-initdb --reset` | Wipe the index and rebuild it empty, **keeping the cached embeddings** -- a full re-import then costs no re-embedding |
| `cc-initdb --reset-hard` | Drop the database outright, cache included; the next `cc-embed` recomputes every vector |
| `cc-import FILE [--reimport]` | Import a claude.ai export `.zip` into the database (merges; never deletes) |
| `cc-embed` | Embed any prose with no vector yet (resumable) |
| `cc-status` | Show counts (conversations / messages / embedded) |
| `cc-web` | Run the Flask UI |

Run `cc-import` on each new export as it arrives, then `cc-embed`. Both are
incremental: only conversations that actually changed are rebuilt, and vectors are
cached by content, so re-embedding happens only for prose that is genuinely new.
