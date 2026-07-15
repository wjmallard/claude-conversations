"""Local MLX embedding engine for semantic search (Apple Silicon, no API cost).

Adapted from the twitter-news project: same model and Qwen3-Embedding last-token
pooling + L2 normalization. Imported lazily so the base tool runs without the
heavy `semantic` extras installed.
"""

import logging
import sys

import mlx.core as mx
from mlx_lm import load as mlx_load
from tqdm import tqdm

from claude_conversations import config
from claude_conversations.db import get_conn

log = logging.getLogger(__name__)

# Fixed total token budget per batch keeps transformer memory roughly constant:
# batch_size = _MAX_TOKENS / tokens_per_item. Also the truncation ceiling.
_MAX_TOKENS = 8192
_MAX_BATCH_SIZE = 32

_model = None
_tokenizer = None


def load_model():
    """Load the embedding model + tokenizer into module state (no-op if loaded)."""
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"Loading embedding model {config.EMBEDDING_MODEL_ID}...", file=sys.stderr)
    _model, _tokenizer = mlx_load(config.EMBEDDING_MODEL_ID)
    mx.eval(_model.parameters())
    print("Embedding model ready.", file=sys.stderr)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Last-token pooling + L2 norm (Qwen3 convention)."""
    tokens = _tokenizer._tokenizer(
        texts, return_tensors="np", padding=True, truncation=True, max_length=_MAX_TOKENS,
    )
    input_ids = mx.array(tokens["input_ids"])
    attention_mask = mx.array(tokens["attention_mask"])

    hidden = _model.model(input_ids)  # transformer body, skip LM head

    seq_lengths = attention_mask.sum(axis=1) - 1  # last non-pad position
    batch_idx = mx.arange(hidden.shape[0])
    embeds = hidden[batch_idx, seq_lengths]

    norms = mx.linalg.norm(embeds, axis=1, keepdims=True)
    embeds = embeds / mx.where(norms == 0, 1, norms)

    mx.eval(embeds)
    result = embeds.tolist()
    del hidden, embeds, input_ids, attention_mask
    mx.clear_cache()
    return result


def vec_literal(embedding: list[float]) -> str:
    """Format a float list as a pgvector text literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


def _adaptive_batch_size(text_len: int) -> int:
    estimated_tokens = max(1, text_len // 4)
    return max(1, min(_MAX_BATCH_SIZE, _MAX_TOKENS // estimated_tokens))


def backfill_embeddings() -> int:
    """Embed every distinct chunk text that has no vector yet. Returns count embedded.

    Vectors are keyed by the sha256 of their text, not by a chunk row, so re-indexing a
    conversation -- which deletes and recreates its chunks -- costs nothing here: the
    text is unchanged, its hash is unchanged, and the vector is already cached. Only
    genuinely new prose is computed.
    """
    with get_conn() as conn:
        # Shortest first, so _adaptive_batch_size packs each batch with similarly sized
        # texts. text_len is in the select list because SELECT DISTINCT can only order
        # by expressions it selects; it does not affect distinctness, being a function
        # of text.
        rows = conn.execute("""
            SELECT DISTINCT
                c.text_sha256,
                c.text,
                length(c.text) AS text_len
            FROM message_chunks c
            LEFT JOIN embeddings e ON e.text_sha256 = c.text_sha256
            WHERE e.text_sha256 IS NULL
              AND c.text <> ''
            ORDER BY text_len
        """).fetchall()
        if not rows:
            print("Nothing to embed -- every chunk's text already has a vector.", file=sys.stderr)
            return 0

        load_model()
        items = [(r["text_sha256"], r["text"]) for r in rows]

        progress = tqdm(total=len(items), desc="Embedding chunks", file=sys.stderr)
        i = 0
        try:
            while i < len(items):
                batch_size = _adaptive_batch_size(len(items[i][1]))
                batch = items[i:i + batch_size]
                digests = [p[0] for p in batch]
                texts = [p[1] for p in batch]

                embeddings = embed_texts(texts)
                for digest, emb in zip(digests, embeddings):
                    conn.execute(
                        """
                        INSERT INTO embeddings (
                            text_sha256,
                            embedding
                        )
                        VALUES (
                            %(text_sha256)s,
                            %(vec)s::vector
                        )
                        ON CONFLICT (text_sha256) DO NOTHING
                        """,
                        {
                            "text_sha256": digest,
                            "vec": vec_literal(emb),
                        },
                    )
                conn.commit()
                progress.update(len(batch))
                i += batch_size
        except KeyboardInterrupt:
            conn.commit()
            print("\nInterrupted -- progress saved; rerun cc-embed to continue.", file=sys.stderr)
        finally:
            progress.close()

    return len(items)
