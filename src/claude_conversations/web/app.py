"""Flask web UI: browse, search (keyword / fuzzy / semantic), and read conversations."""

import json
import re
from datetime import date, datetime

from flask import Flask, Response, abort, redirect, render_template, request, url_for
from markupsafe import Markup, escape

from claude_conversations import categories, classifier, config, parse, render
from claude_conversations.db import (
    HL_START,
    HL_STOP,
    category_facets,
    check_db,
    conversations_by_uuids,
    count_conversations,
    count_fulltext,
    count_semantic,
    count_trigram,
    get_conn,
    get_conversation,
    list_conversations,
    search_fulltext,
    search_semantic,
    search_trigram,
    set_conversation_categories,
)

PER_PAGE = 50
REVIEW_PER_PAGE = 30
MODES = {"word", "char", "semantic"}
ALL_FIELDS = ["title", "summary", "you", "assistant", "tools"]
FIELDS_BY_MODE = {
    "word": ALL_FIELDS,
    "char": ALL_FIELDS,
    "semantic": ["you", "assistant"],
}
DEFAULT_FIELDS = ["you", "assistant"]
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

app = Flask(__name__)

_embed_loaded = False


def _resolve_fields(mode):
    """(selected_fields, applicable_fields) for a search mode, from request args.
    Defaults to your + assistant prose; non-applicable fields are dropped."""
    applicable = FIELDS_BY_MODE.get(mode, ALL_FIELDS)
    fields = [f for f in (request.args.getlist("field") or DEFAULT_FIELDS) if f in applicable]
    if not fields:
        fields = [f for f in DEFAULT_FIELDS if f in applicable]
    return fields, applicable


@app.context_processor
def _inject_categories():
    """Category vocabulary, search-field state, and tag provenance, for all templates."""
    mode = request.args.get("mode", "word")
    if mode not in MODES:
        mode = "word"
    search_fields, applicable = _resolve_fields(mode)
    data = categories.load()
    _, proposal_counts = categories.proposal_queue(data)
    return {
        "categories_vocab": sorted(config.CATEGORIES, key=lambda c: c.get("label", c["slug"]).lower()),
        "category_label": {c["slug"]: c.get("label", c["slug"]) for c in config.CATEGORIES},
        "search_fields": search_fields,
        "search_fields_applicable": applicable,
        "tag_methods": categories.method_map(data),
        "review_count": sum(proposal_counts.values()),
    }


def _get_query_vec(text):
    """Embed a query string for semantic search. Lazy-loads the MLX model once.
    (Imported inside the function so the base tool runs without the semantic extras.)"""
    global _embed_loaded
    from claude_conversations.embedding import embed_texts, load_model, vec_literal
    if not _embed_loaded:
        load_model()
        _embed_loaded = True
    return vec_literal(embed_texts([text])[0])


@app.template_filter("fmt_ts")
def _fmt_ts(v):
    if not v:
        return ""
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return v
    try:
        return v.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(v)


@app.template_filter("highlight")
def _highlight(s):
    """Escape a snippet, then turn ts_headline sentinels into <mark> tags."""
    if not s:
        return ""
    out = escape(s)
    return out.replace(HL_START, Markup("<mark>")).replace(HL_STOP, Markup("</mark>"))


def _page_numbers(current, total):
    """Compact pagination: 1 2 3 ... 9 10."""
    if total <= 7:
        return list(range(1, total + 1))
    if current <= 4:
        return [1, 2, 3, 4, 5, None, total]
    if current >= total - 3:
        return [1, None, total - 4, total - 3, total - 2, total - 1, total]
    return [1, None, current - 1, current, current + 1, None, total]


def _parse_date(s):
    """Parse a YYYY-MM-DD string into a date, or None if absent/invalid."""
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    mode = request.args.get("mode", "word")
    if mode not in MODES:
        mode = "word"
    sort = request.args.get("sort") or ("best" if q else "recent")
    category = request.args.get("category") or None
    df = _parse_date(request.args.get("df"))
    dt = _parse_date(request.args.get("dt"))
    fields, _ = _resolve_fields(mode)
    page = max(1, request.args.get("page", 1, type=int))
    offset = (page - 1) * PER_PAGE

    with get_conn() as conn:
        total_indexed = count_conversations(conn)
        facets, untagged = category_facets(conn)
        if q:
            if mode == "semantic":
                vec = _get_query_vec(q)
                floor = config.EMBEDDING_SIMILARITY_FLOOR
                results = search_semantic(
                    conn,
                    vec,
                    PER_PAGE,
                    offset,
                    sort,
                    min_score=floor,
                    date_from=df,
                    date_to=dt,
                    fields=fields,
                )
                total = count_semantic(
                    conn,
                    vec,
                    min_score=floor,
                    date_from=df,
                    date_to=dt,
                    fields=fields,
                )
            elif mode == "char":
                results = search_trigram(
                    conn,
                    q,
                    PER_PAGE,
                    offset,
                    sort,
                    date_from=df,
                    date_to=dt,
                    fields=fields,
                )
                total = count_trigram(
                    conn,
                    q,
                    date_from=df,
                    date_to=dt,
                    fields=fields,
                )
            else:
                results = search_fulltext(
                    conn,
                    q,
                    PER_PAGE,
                    offset,
                    sort,
                    date_from=df,
                    date_to=dt,
                    fields=fields,
                )
                total = count_fulltext(
                    conn,
                    q,
                    date_from=df,
                    date_to=dt,
                    fields=fields,
                )
        else:
            results = list_conversations(
                conn,
                PER_PAGE,
                offset,
                sort,
                category=category,
                date_from=df,
                date_to=dt,
            )
            total = count_conversations(
                conn,
                category=category,
                date_from=df,
                date_to=dt,
            )

    total_pages = (total + PER_PAGE - 1) // PER_PAGE if total else 0
    return render_template(
        "list.html",
        q=q,
        mode=mode,
        sort=sort,
        category=category,
        df=(df.isoformat() if df else ""),
        dt=(dt.isoformat() if dt else ""),
        results=results,
        total=total,
        total_indexed=total_indexed,
        facets=facets,
        untagged=untagged,
        page=page,
        total_pages=total_pages,
        pages=_page_numbers(page, total_pages),
    )


@app.route("/c/<uuid>")
def conversation(uuid):
    if not _UUID_RE.match(uuid):
        abort(404)
    with get_conn() as conn:
        meta = get_conversation(conn, uuid)
    if not meta:
        abort(404)
    jsonl = config.CONVERSATIONS_DIR / (uuid + ".jsonl")
    messages = [render.render_message(m) for m in parse.load_messages(jsonl)]
    return render_template("detail.html", meta=meta, messages=messages, uuid=uuid)


@app.route("/c/<uuid>/tags", methods=["POST"])
def set_tags(uuid):
    """Save a conversation's category tags (a manual edit): authoritative to the
    curation file, mirrored into the DB. Marks the conversation user-locked."""
    if not _UUID_RE.match(uuid):
        abort(404)
    valid = set(config.CATEGORY_SLUGS)
    slugs = [s for s in request.form.getlist("category") if s in valid]
    data = categories.load()
    categories.set_user_tags(data, uuid, slugs)
    categories.save(data)
    with get_conn() as conn:
        set_conversation_categories(conn, uuid, slugs)
    if request.form.get("ajax"):
        return {"ok": True, "uuid": uuid, "slugs": sorted(slugs)}
    return redirect(request.form.get("next") or url_for("conversation", uuid=uuid))


@app.route("/review")
def review():
    """Queue of semantic proposals for fast inline curation (chips + Confirm/Dismiss),
    filterable by category, a message-free conversation date range, and a title query."""
    category = request.args.get("category") or None
    q = request.args.get("q", "").strip()
    df = _parse_date(request.args.get("df"))
    dt = _parse_date(request.args.get("dt"))
    page = max(1, request.args.get("page", 1, type=int))
    data = categories.load()
    queue, counts = categories.proposal_queue(data)
    if category:
        queue = [entry for entry in queue if category in entry["proposed"]]
    with get_conn() as conn:
        meta = conversations_by_uuids(conn, [entry["uuid"] for entry in queue])

    def keep(entry):
        m = meta.get(entry["uuid"])
        if not m:
            return False
        day = m["created_at"].date() if m["created_at"] else None
        if df and (day is None or day < df):
            return False
        if dt and (day is None or day > dt):
            return False
        if q and q.lower() not in (m["name"] or "").lower():
            return False
        return True

    queue = [entry for entry in queue if keep(entry)]
    total = len(queue)
    total_pages = (total + REVIEW_PER_PAGE - 1) // REVIEW_PER_PAGE if total else 0
    if total_pages:
        page = min(page, total_pages)
    offset = (page - 1) * REVIEW_PER_PAGE
    cards = [dict(entry, meta=meta[entry["uuid"]]) for entry in queue[offset : offset + REVIEW_PER_PAGE]]
    return render_template(
        "review.html",
        cards=cards,
        counts=counts,
        category=category,
        q=q,
        df=(df.isoformat() if df else ""),
        dt=(dt.isoformat() if dt else ""),
        total=total,
        page=page,
        total_pages=total_pages,
        pages=_page_numbers(page, total_pages),
    )


@app.route("/review/rerun", methods=["POST"])
def review_rerun():
    """Recompute the semantic proposals from the current seeds (incl. confirmations)."""
    classifier.semantic_pass(dry_run=False, verbose=False)
    return redirect(url_for("review", category=request.form.get("category") or None))


@app.route("/raw/<uuid>")
def raw(uuid):
    if not _UUID_RE.match(uuid):
        abort(404)
    jsonl = config.CONVERSATIONS_DIR / (uuid + ".jsonl")
    msgs = parse.load_messages(jsonl)
    if not msgs:
        abort(404)
    return Response(
        json.dumps(msgs, indent=2, ensure_ascii=False),
        mimetype="text/plain; charset=utf-8",
    )


def main():
    check_db()
    app.run(debug=True, port=config.FLASK_PORT)
