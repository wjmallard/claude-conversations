"""Database connection and queries.

Search is FIELD-SCOPED: each mode searches a selected set of fields --
  title / summary / you (human prose) / assistant (assistant prose) / tools (tool I/O)
-- defaulting to your + assistant prose. Keyword/fuzzy support all five; semantic
only supports the embedded prose fields (you/assistant). A message-level date
window further restricts the message-derived fields (you/assistant/tools); title
and summary are conversation-level and not date-filtered.

All modes operate on individual messages/fields, then aggregate the best score
per conversation and return conversation rows.

Note on psycopg + pg_trgm: the `<<%` word-similarity operator must be written
`<<%%` in the query string because psycopg treats `%` as a parameter marker.
"""

import sys
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from claude_conversations import config

DB_NAME = config.DB_NAME

HL_START = "\x02"
HL_STOP = "\x03"
_HEADLINE_OPTS = (
    f"StartSel={HL_START}, StopSel={HL_STOP}, "
    "MaxFragments=2, MaxWords=22, MinWords=8, FragmentDelimiter= … "
)

_SEARCH_SORTS = {
    "best": "score DESC, c.updated_at DESC NULLS LAST",
    "newest": "c.created_at DESC NULLS LAST",
    "oldest": "c.created_at ASC NULLS LAST",
    "title": "c.name ASC NULLS LAST",
}

_BROWSE_SORTS = {
    "recent": "updated_at DESC NULLS LAST",
    "created": "created_at DESC NULLS LAST",
    "oldest": "created_at ASC NULLS LAST",
    "title": "name ASC NULLS LAST",
    "messages": "n_messages DESC",
}

UNTAGGED = "__untagged__"          # category-filter sentinel: conversations with no tags
ALL_FIELDS = ("title", "summary", "you", "assistant", "tools")
DEFAULT_FIELDS = ("you", "assistant")
_TOOL_CAP = 500000                  # cap tool_text length fed to tsvector/word_similarity


def _search_order(sort):
    return _SEARCH_SORTS.get(sort, _SEARCH_SORTS["best"])


def _msg_date_clause(date_from, date_to, alias="m"):
    """SQL fragment (leading ' AND ...') + params restricting <alias>.created_at to
    [date_from, date_to]; date_to is inclusive of its whole day. Empty if no window."""
    clauses, params = [], {}
    if date_from:
        clauses.append(f"{alias}.created_at >= %(df)s")
        params["df"] = date_from
    if date_to:
        clauses.append(f"{alias}.created_at < (%(dt)s::date + 1)")
        params["dt"] = date_to
    return "".join(" AND " + c for c in clauses), params


def _prose_sender(fields):
    """(include_prose, sender_sql) for the message-prose part, per you/assistant."""
    you, asst = "you" in fields, "assistant" in fields
    if you and asst:
        return True, ""
    if you:
        return True, " AND m.sender = 'human'"
    if asst:
        return True, " AND m.sender = 'assistant'"
    return False, ""


def check_db():
    """Verify PostgreSQL + the target database are reachable. Call once at startup."""
    try:
        psycopg.connect(dbname=DB_NAME).close()
    except psycopg.OperationalError as exc:
        print(f"Error: could not connect to PostgreSQL database {DB_NAME!r}: {exc}", file=sys.stderr)
        print("Is the server running, and have you run `cc-initdb`?", file=sys.stderr)
        sys.exit(1)


@contextmanager
def get_conn():
    conn = psycopg.connect(dbname=DB_NAME, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_schema(conn):
    """(Re)create the schema from sql/schema.sql."""
    schema_path = config._PROJECT_ROOT / "sql" / "schema.sql"
    conn.execute(schema_path.read_text())


# ---------------------------------------------------------------- search


def search_fulltext(
    conn,
    query,
    limit=50,
    offset=0,
    sort="best",
    date_from=None,
    date_to=None,
    fields=None,
):
    fields = set(fields) if fields else set(DEFAULT_FIELDS)
    order = _search_order(sort)
    dclause, params = _msg_date_clause(date_from, date_to, "m")
    inc, sender = _prose_sender(fields)
    parts = []
    if inc:
        parts.append(f"""
            SELECT
                m.conv_uuid,
                m.text AS snip_src,
                ts_rank(m.text_tsv, q.tsq) AS r
            FROM messages m, q
            WHERE m.text_tsv @@ q.tsq{sender}{dclause}""")
    if "tools" in fields:
        parts.append(f"""
            SELECT
                m.conv_uuid,
                left(m.tool_text, {_TOOL_CAP}) AS snip_src,
                ts_rank(to_tsvector('english', left(m.tool_text, {_TOOL_CAP})), q.tsq) AS r
            FROM messages m, q
            WHERE m.tool_text IS NOT NULL
              AND to_tsvector('english', left(m.tool_text, {_TOOL_CAP})) @@ q.tsq{dclause}""")
    if "title" in fields:
        parts.append("""
            SELECT
                c.uuid AS conv_uuid,
                c.name AS snip_src,
                ts_rank(to_tsvector('english', c.name), q.tsq) * 3 AS r
            FROM conversations c, q
            WHERE c.name IS NOT NULL
              AND c.name <> ''
              AND to_tsvector('english', c.name) @@ q.tsq""")
    if "summary" in fields:
        parts.append("""
            SELECT
                c.uuid AS conv_uuid,
                c.summary AS snip_src,
                ts_rank(to_tsvector('english', c.summary), q.tsq) * 2 AS r
            FROM conversations c, q
            WHERE c.summary IS NOT NULL
              AND c.summary <> ''
              AND to_tsvector('english', c.summary) @@ q.tsq""")
    if not parts:
        return []
    params.update({
        "query": query,
        "opts": _HEADLINE_OPTS,
        "limit": limit,
        "offset": offset,
    })
    return conn.execute(
        f"""
        WITH q AS (SELECT websearch_to_tsquery('english', %(query)s) AS tsq),
        all_hits AS ({" UNION ALL ".join(parts)}),
        best AS (
            SELECT
                conv_uuid,
                max(r) AS score,
                (array_agg(snip_src ORDER BY r DESC))[1] AS best_text
            FROM all_hits
            GROUP BY conv_uuid
        )
        SELECT
            c.uuid,
            c.name,
            c.created_at,
            c.updated_at,
            c.n_messages,
            c.categories,
            b.score,
            ts_headline('english', b.best_text, (SELECT tsq FROM q), %(opts)s) AS snippet
        FROM best b
        JOIN conversations c ON c.uuid = b.conv_uuid
        ORDER BY {order}
        LIMIT %(limit)s OFFSET %(offset)s
        """,
        params,
    ).fetchall()


def count_fulltext(conn, query, date_from=None, date_to=None, fields=None):
    fields = set(fields) if fields else set(DEFAULT_FIELDS)
    dclause, params = _msg_date_clause(date_from, date_to, "m")
    inc, sender = _prose_sender(fields)
    parts = []
    if inc:
        parts.append(f"""
            SELECT m.conv_uuid AS conv_uuid
            FROM messages m, q
            WHERE m.text_tsv @@ q.tsq{sender}{dclause}""")
    if "tools" in fields:
        parts.append(f"""
            SELECT m.conv_uuid AS conv_uuid
            FROM messages m, q
            WHERE m.tool_text IS NOT NULL
              AND to_tsvector('english', left(m.tool_text, {_TOOL_CAP})) @@ q.tsq{dclause}""")
    if "title" in fields:
        parts.append("""
            SELECT c.uuid AS conv_uuid
            FROM conversations c, q
            WHERE c.name IS NOT NULL
              AND c.name <> ''
              AND to_tsvector('english', c.name) @@ q.tsq""")
    if "summary" in fields:
        parts.append("""
            SELECT c.uuid AS conv_uuid
            FROM conversations c, q
            WHERE c.summary IS NOT NULL
              AND c.summary <> ''
              AND to_tsvector('english', c.summary) @@ q.tsq""")
    if not parts:
        return 0
    params["query"] = query
    return conn.execute(
        f"""
        WITH q AS (SELECT websearch_to_tsquery('english', %(query)s) AS tsq),
        hits AS ({" UNION ".join(parts)})
        SELECT count(DISTINCT conv_uuid)
        FROM hits
        """,
        params,
    ).fetchone()["count"]


def search_trigram(
    conn,
    query,
    limit=50,
    offset=0,
    sort="best",
    date_from=None,
    date_to=None,
    fields=None,
):
    fields = set(fields) if fields else set(DEFAULT_FIELDS)
    order = _search_order(sort)
    dclause, params = _msg_date_clause(date_from, date_to, "m")
    inc, sender = _prose_sender(fields)
    parts = []
    if inc:
        parts.append(f"""
            SELECT
                m.conv_uuid,
                m.text AS snip_src,
                word_similarity(%(query)s, m.text) AS r
            FROM messages m
            WHERE %(query)s <<%% m.text{sender}{dclause}""")
    if "tools" in fields:
        parts.append(f"""
            SELECT
                m.conv_uuid,
                left(m.tool_text, {_TOOL_CAP}) AS snip_src,
                word_similarity(%(query)s, left(m.tool_text, {_TOOL_CAP})) AS r
            FROM messages m
            WHERE m.tool_text IS NOT NULL
              AND %(query)s <<%% left(m.tool_text, {_TOOL_CAP}){dclause}""")
    if "title" in fields:
        parts.append("""
            SELECT
                c.uuid AS conv_uuid,
                c.name AS snip_src,
                word_similarity(%(query)s, c.name) * 1.5 AS r
            FROM conversations c
            WHERE c.name IS NOT NULL
              AND %(query)s <<%% c.name""")
    if "summary" in fields:
        parts.append("""
            SELECT
                c.uuid AS conv_uuid,
                c.summary AS snip_src,
                word_similarity(%(query)s, c.summary) AS r
            FROM conversations c
            WHERE c.summary IS NOT NULL
              AND %(query)s <<%% c.summary""")
    if not parts:
        return []
    params.update({
        "query": query,
        "limit": limit,
        "offset": offset,
    })
    return conn.execute(
        f"""
        WITH all_hits AS ({" UNION ALL ".join(parts)}),
        best AS (
            SELECT
                conv_uuid,
                max(r) AS score,
                (array_agg(snip_src ORDER BY r DESC))[1] AS best_text
            FROM all_hits
            GROUP BY conv_uuid
        )
        SELECT
            c.uuid,
            c.name,
            c.created_at,
            c.updated_at,
            c.n_messages,
            c.categories,
            b.score,
            left(b.best_text, 300) AS snippet
        FROM best b
        JOIN conversations c ON c.uuid = b.conv_uuid
        ORDER BY {order}
        LIMIT %(limit)s OFFSET %(offset)s
        """,
        params,
    ).fetchall()


def count_trigram(conn, query, date_from=None, date_to=None, fields=None):
    fields = set(fields) if fields else set(DEFAULT_FIELDS)
    dclause, params = _msg_date_clause(date_from, date_to, "m")
    inc, sender = _prose_sender(fields)
    parts = []
    if inc:
        parts.append(f"""
            SELECT m.conv_uuid AS conv_uuid
            FROM messages m
            WHERE %(query)s <<%% m.text{sender}{dclause}""")
    if "tools" in fields:
        parts.append(f"""
            SELECT m.conv_uuid AS conv_uuid
            FROM messages m
            WHERE m.tool_text IS NOT NULL
              AND %(query)s <<%% left(m.tool_text, {_TOOL_CAP}){dclause}""")
    if "title" in fields:
        parts.append("""
            SELECT c.uuid AS conv_uuid
            FROM conversations c
            WHERE c.name IS NOT NULL
              AND %(query)s <<%% c.name""")
    if "summary" in fields:
        parts.append("""
            SELECT c.uuid AS conv_uuid
            FROM conversations c
            WHERE c.summary IS NOT NULL
              AND %(query)s <<%% c.summary""")
    if not parts:
        return 0
    params["query"] = query
    return conn.execute(
        f"""
        WITH hits AS ({" UNION ".join(parts)})
        SELECT count(DISTINCT conv_uuid)
        FROM hits
        """,
        params,
    ).fetchone()["count"]


def search_semantic(
    conn,
    query_vec,
    limit=50,
    offset=0,
    sort="best",
    min_score=0.4,
    date_from=None,
    date_to=None,
    fields=None,
):
    fields = set(fields) if fields else set(DEFAULT_FIELDS)
    order = _search_order(sort)
    dclause, params = _msg_date_clause(date_from, date_to, "m")
    inc, sender = _prose_sender(fields)
    if not inc:  # semantic only supports prose (you/assistant)
        return []
    params.update({
        "vec": query_vec,
        "min_score": min_score,
        "limit": limit,
        "offset": offset,
    })
    return conn.execute(
        f"""
        WITH scores AS (
            SELECT
                m.conv_uuid,
                m.text,
                1 - (m.embedding <=> %(vec)s::vector) AS s
            FROM message_chunks m
            WHERE m.embedding IS NOT NULL{sender}{dclause}
        ),
        best AS (
            SELECT
                conv_uuid,
                max(s) AS score,
                (array_agg(text ORDER BY s DESC))[1] AS best_text
            FROM scores
            WHERE s >= %(min_score)s
            GROUP BY conv_uuid
        )
        SELECT
            c.uuid,
            c.name,
            c.created_at,
            c.updated_at,
            c.n_messages,
            c.categories,
            b.score,
            left(b.best_text, 300) AS snippet
        FROM best b
        JOIN conversations c ON c.uuid = b.conv_uuid
        ORDER BY {order}
        LIMIT %(limit)s OFFSET %(offset)s
        """,
        params,
    ).fetchall()


def count_semantic(conn, query_vec, min_score=0.4, date_from=None, date_to=None, fields=None):
    fields = set(fields) if fields else set(DEFAULT_FIELDS)
    dclause, params = _msg_date_clause(date_from, date_to, "m")
    inc, sender = _prose_sender(fields)
    if not inc:
        return 0
    params.update({
        "vec": query_vec,
        "min_score": min_score,
    })
    return conn.execute(
        f"""
        WITH scores AS (
            SELECT
                m.conv_uuid,
                max(1 - (m.embedding <=> %(vec)s::vector)) AS s
            FROM message_chunks m
            WHERE m.embedding IS NOT NULL{sender}{dclause}
            GROUP BY m.conv_uuid
        )
        SELECT count(*)
        FROM scores
        WHERE s >= %(min_score)s
        """,
        params,
    ).fetchone()["count"]


def semantic_centroid_scores(conn, seeds, exclude):
    """Cosine similarity of each candidate conversation's centroid to each category
    centroid, for the semantic classifier (layer 2).

      conversation centroid = mean of the conversation's prose-message embeddings;
      category centroid      = mean of the conversation centroids of its seed
                               conversations (the current keyword/topic/user tags).

    `seeds` maps slug -> list of seed conv uuids; `exclude` is the set of uuids NOT
    to score as candidates (the seed conversations plus user-locked ones). Returns
    rows {uuid, name, created_at, slug, sim} for every (candidate, category) pair --
    candidates are conversations that have an embedding and are not in `exclude`."""
    seed_uuids, seed_slugs = [], []
    for slug, uuids in seeds.items():
        for u in uuids:
            seed_uuids.append(u)
            seed_slugs.append(slug)
    if not seed_uuids:
        return []
    return conn.execute(
        """
        WITH seed (uuid, slug) AS (
            SELECT *
            FROM unnest(%(seed_uuids)s::text[], %(seed_slugs)s::text[])
        ),
        conv_centroids AS (
            SELECT
                m.conv_uuid AS uuid,
                avg(m.embedding) AS centroid
            FROM message_chunks m
            WHERE m.embedding IS NOT NULL
            GROUP BY m.conv_uuid
        ),
        cat_centroids AS (
            SELECT
                s.slug,
                avg(cc.centroid) AS centroid
            FROM seed s
            JOIN conv_centroids cc ON cc.uuid = s.uuid
            GROUP BY s.slug
        )
        SELECT
            cc.uuid,
            c.name,
            c.created_at,
            cat.slug,
            1 - (cc.centroid <=> cat.centroid) AS sim
        FROM conv_centroids cc
        JOIN conversations c ON c.uuid = cc.uuid
        CROSS JOIN cat_centroids cat
        WHERE cc.uuid <> ALL(%(exclude)s::text[])
        """,
        {
            "seed_uuids": seed_uuids,
            "seed_slugs": seed_slugs,
            "exclude": list(exclude),
        },
    ).fetchall()


def centroid_scores_for(conn, seeds, target_uuids):
    """Cosine similarity of each target conversation's centroid to every category
    centroid -- for showing all plausible categories (not just the top pick) in the
    review UI. Centroids are computed only over the seed + target conversations, so
    this stays cheap enough to run per page load. `seeds` maps slug -> seed conv
    uuids. Returns rows {uuid, created_at, slug, sim}."""
    seed_uuids, seed_slugs = [], []
    for slug, uuids in seeds.items():
        for u in uuids:
            seed_uuids.append(u)
            seed_slugs.append(slug)
    targets = list(target_uuids)
    if not seed_uuids or not targets:
        return []
    relevant = list(set(seed_uuids) | set(targets))
    return conn.execute(
        """
        WITH seed (uuid, slug) AS (
            SELECT *
            FROM unnest(%(seed_uuids)s::text[], %(seed_slugs)s::text[])
        ),
        conv_centroids AS (
            SELECT
                m.conv_uuid AS uuid,
                avg(m.embedding) AS centroid
            FROM message_chunks m
            WHERE m.embedding IS NOT NULL
              AND m.conv_uuid = ANY(%(relevant)s::text[])
            GROUP BY m.conv_uuid
        ),
        cat_centroids AS (
            SELECT
                s.slug,
                avg(cc.centroid) AS centroid
            FROM seed s
            JOIN conv_centroids cc ON cc.uuid = s.uuid
            GROUP BY s.slug
        )
        SELECT
            cc.uuid,
            c.created_at,
            cat.slug,
            1 - (cc.centroid <=> cat.centroid) AS sim
        FROM conv_centroids cc
        JOIN conversations c ON c.uuid = cc.uuid
        CROSS JOIN cat_centroids cat
        WHERE cc.uuid = ANY(%(targets)s::text[])
        """,
        {
            "seed_uuids": seed_uuids,
            "seed_slugs": seed_slugs,
            "relevant": relevant,
            "targets": targets,
        },
    ).fetchall()


# ---------------------------------------------------------------- browse / meta


def _browse_where(category=None, date_from=None, date_to=None):
    """(WHERE clause, params) for browsing conversations: combines a category facet
    with a message-level date window (conversations having a message in range)."""
    conds, params = [], {}
    if category == UNTAGGED:
        conds.append("categories = '{}'::text[]")
    elif category:
        conds.append("categories @> ARRAY[%(cat)s]")
        params["cat"] = category
    if date_from or date_to:
        dclause, dparams = _msg_date_clause(date_from, date_to, "m")
        conds.append(
            f"EXISTS (SELECT 1 FROM messages m WHERE m.conv_uuid = conversations.uuid{dclause})"
        )
        params.update(dparams)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return where, params


def list_conversations(
    conn,
    limit=50,
    offset=0,
    sort="recent",
    category=None,
    date_from=None,
    date_to=None,
):
    order = _BROWSE_SORTS.get(sort, _BROWSE_SORTS["recent"])
    where, params = _browse_where(category, date_from, date_to)
    params.update({
        "limit": limit,
        "offset": offset,
    })
    return conn.execute(
        f"""
        SELECT
            uuid,
            name,
            created_at,
            updated_at,
            n_messages,
            categories,
            NULL::float8 AS score,
            NULL::text AS snippet
        FROM conversations
        {where}
        ORDER BY {order}
        LIMIT %(limit)s OFFSET %(offset)s
        """,
        params,
    ).fetchall()


def get_conversation(conn, uuid):
    return conn.execute(
        "SELECT * FROM conversations WHERE uuid = %(uuid)s",
        {"uuid": uuid},
    ).fetchone()


def conversations_by_uuids(conn, uuids):
    """Look up conversation rows for a list of uuids, keyed by uuid (review queue)."""
    if not uuids:
        return {}
    rows = conn.execute(
        """
        SELECT
            uuid,
            name,
            created_at,
            n_messages,
            categories
        FROM conversations
        WHERE uuid = ANY(%(uuids)s)
        """,
        {"uuids": list(uuids)},
    ).fetchall()
    return {r["uuid"]: r for r in rows}


def count_conversations(conn, category=None, date_from=None, date_to=None):
    where, params = _browse_where(category, date_from, date_to)
    return conn.execute(
        f"SELECT count(*) FROM conversations {where}",
        params,
    ).fetchone()["count"]


def category_facets(conn):
    """Counts per category slug, plus the untagged count, for the filter UI."""
    rows = conn.execute(
        "SELECT unnest(categories) AS slug, count(*) FROM conversations GROUP BY 1"
    ).fetchall()
    facets = {r["slug"]: r["count"] for r in rows}
    untagged = conn.execute(
        "SELECT count(*) FROM conversations WHERE categories = '{}'::text[]"
    ).fetchone()["count"]
    return facets, untagged


def set_conversation_categories(conn, uuid, slugs):
    """Mirror a single conversation's category tags into the DB column."""
    conn.execute(
        "UPDATE conversations SET categories = %(c)s WHERE uuid = %(u)s",
        {
            "c": sorted(set(slugs)),
            "u": uuid,
        },
    )


def stats(conn):
    return conn.execute(
        """
        SELECT
            (SELECT count(*) FROM conversations) AS conversations,
            (SELECT count(*) FROM messages) AS messages,
            (SELECT count(*) FROM message_chunks) AS chunks,
            (SELECT count(*) FROM message_chunks WHERE embedding IS NOT NULL) AS embedded
        """
    ).fetchone()
