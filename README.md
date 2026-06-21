# claude-conversations

An offline browser and search engine for exported Claude conversations.

- **Browse** conversations, sorted by date / title / length.
- **Search** with three modes:
  - **keyword** — Postgres full-text (`tsvector`), stemmed, with highlighted snippets
  - **fuzzy** — `pg_trgm` trigram similarity, typo- and substring-tolerant
  - **semantic** — local MLX embeddings (Qwen3) + `pgvector` cosine, concept search
- **Read** a conversation with Markdown rendering and collapsible thinking / tool-call / tool-result blocks.

## Setup

Requires PostgreSQL with the `pg_trgm` and `vector` extensions, and (for semantic
search) Apple Silicon for the local MLX embedder.

```sh
cp config.yaml.example config.yaml   # then edit conversations_dir / db_name
uv sync                              # base deps (keyword + fuzzy + browse)
uv run cc-initdb                     # create database + schema
uv run cc-index                      # index the conversations (incremental)
uv run cc-web                        # serve at http://127.0.0.1:5005
```

### Semantic search (optional)

```sh
uv sync --extra semantic             # mlx + mlx-lm
uv run cc-embed                      # embed messages locally
```

Then select **semantic** mode in the search bar.

## Commands

| Command | What it does |
| --- | --- |
| `cc-initdb [--reset]` | Create the database and apply `sql/schema.sql` (`--reset` drops tables first) |
| `cc-index [--reindex]` | Index conversations; incremental by file mtime/size (`--reindex` forces all) |
| `cc-embed` | Embed messages with no embedding yet (resumable) |
| `cc-status` | Show counts (conversations / messages / embedded) |
| `cc-web` | Run the Flask UI |

Re-run `cc-index` (and `cc-embed`) after adding new exports — both are incremental.
