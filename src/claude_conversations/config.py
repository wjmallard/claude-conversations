"""Load project config from config.yaml (and optional .env)."""

from pathlib import Path

from dotenv import load_dotenv
import yaml


def _find_project_root():
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml)")


_PROJECT_ROOT = _find_project_root()

load_dotenv(_PROJECT_ROOT / ".env")

_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"{_CONFIG_PATH} not found. Copy config.yaml.example to config.yaml and edit it."
    )

with open(_CONFIG_PATH) as f:
    _raw = yaml.safe_load(f) or {}

# Database
DB_NAME = _raw.get("db_name", "claude_conversations")

# Flask
FLASK_PORT = _raw.get("flask_port", 5005)

# Embeddings
EMBEDDING_MODEL_ID = _raw.get("embedding_model_id", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
EMBEDDING_SIMILARITY_FLOOR = _raw.get("embedding_similarity_floor", 0.4)
EMBEDDING_DIM = 1024
# Prose shorter than this is not embedded at all: a bare "." (a hard-fork marker) or
# "yes" carries nothing retrievable, and embedding it only adds noise to search and to
# the category centroids. Keyword and fuzzy search still find it.
EMBEDDING_MIN_CHARS = _raw.get("embedding_min_chars", 12)

# Semantic classifier (layer 2): a conversation is proposed for a category when its
# centroid's cosine similarity to the category centroid clears this threshold; a
# per-category `semantic_threshold` in config.yaml overrides it (a small or
# topically noisy centroid may need a higher bar).
SEMANTIC_THRESHOLD = _raw.get("semantic_threshold", 0.6)

# Category tags. CATEGORIES is the controlled vocabulary (definitions + soft date
# priors + seed terms). The curation file (CATEGORIES_PATH) is the on-disk source
# of truth for assignments; the conversations.categories column mirrors it.
CATEGORIES = _raw.get("categories") or []
CATEGORIES_BY_SLUG = {c["slug"]: c for c in CATEGORIES if c.get("slug")}
CATEGORY_SLUGS = [c["slug"] for c in CATEGORIES if c.get("slug")]

# Classifier routing -- the taxonomy lives here, never in code, so the tool stays
# category-agnostic (and a gitignored config.yaml keeps a private taxonomy private).
# TOPIC_ROUTING drives the layer-1.5 topic pass; DATE_SPLIT_GROUPS tells the semantic
# layer which categories share a topic but are separated by date. Both optional.
TOPIC_ROUTING = _raw.get("topic_routing") or {}
DATE_SPLIT_GROUPS = tuple(
    tuple(group)
    for group in (_raw.get("date_split_groups") or [])
    if group
)

_categories_file = _raw.get("categories_file", "categories.json")
CATEGORIES_PATH = Path(_categories_file).expanduser()
if not CATEGORIES_PATH.is_absolute():
    CATEGORIES_PATH = _PROJECT_ROOT / CATEGORIES_PATH
