"""Sieve classifier.

Layered, cheap-to-expensive assignment of category tags to conversations:

  Layer 1 (keyword) -- match each category's *strong* seed terms (word-boundary,
  case-insensitive) against conversation title + summary. A strong match
  auto-assigns the category (confidence 0.9). For bounded-window categories
  (both date_start and date_end set) matches are hard-gated to the window, so a
  conversation that merely mentions a seed term long after the category's period
  is not tagged. Optional per-category `body` seeds are ultra-distinctive terms
  matched in message bodies (substantial presence) to catch content buried in a
  conversation about another topic.

  Layer 1.5 (topic) -- body-scan a vocabulary shared by sibling categories that are
  separated by date, and route each matched conversation to whichever sibling's
  window contains it, respecting each category's hard window.

The category vocabulary and all routing live in config.yaml -- this module is
taxonomy-agnostic and hardcodes no slugs.

Proposals are written to the curation file via categories.apply_proposal (which
skips user-locked conversations), then mirrored to the DB.
"""

import re
from datetime import date as _date

from claude_conversations import categories, config
from claude_conversations.db import centroid_scores_for, get_conn, semantic_centroid_scores

# Postgres ERE metacharacters to escape inside a seed literal (space and hyphen
# are NOT special and stay as-is so phrases like "cover letter" match).
_PG_META = re.compile(r"([.\\()\[\]{}*+?|^$])")


def _pg_re_escape(s):
    return _PG_META.sub(r"\\\1", s)


def _alt_regex(terms):
    """Word-boundary alternation regex for a list of seed terms, or None."""
    if not terms:
        return None
    return r"\m(" + "|".join(_pg_re_escape(t) for t in terms) + r")\M"


def _strong_match_uuids(conn, rx, scope, date_start=None, date_end=None):
    """Conversation uuids whose text matches `rx`. scope='title' searches the title
    only (highest precision -- for topics like politics whose terms recur as described
    content in other conversations' summaries); scope='meta' searches title + summary
    (what the conversation is *about*); scope='all' also searches message bodies. If
    date_start/date_end are both given (a bounded category window), matches are
    hard-gated to conversations whose created_at is in it."""
    params = {"rx": rx}
    gate = ""
    if date_start and date_end:
        gate = " AND c.created_at >= %(ds)s AND c.created_at < (%(de)s::date + 1)"
        params["ds"] = date_start
        params["de"] = date_end
    if scope == "title":
        rows = conn.execute(
            f"SELECT c.uuid AS u FROM conversations c WHERE c.name ~* %(rx)s{gate}",
            params,
        ).fetchall()
    elif scope == "meta":
        rows = conn.execute(
            f"SELECT c.uuid AS u FROM conversations c"
            f" WHERE (c.name ~* %(rx)s OR c.summary ~* %(rx)s){gate}",
            params,
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT c.uuid AS u
            FROM conversations c
            WHERE (c.name ~* %(rx)s OR c.summary ~* %(rx)s){gate}
            UNION
            SELECT c.uuid AS u
            FROM messages m
            JOIN conversations c ON c.uuid = m.conv_uuid
            WHERE m.text ~* %(rx)s{gate}
            """,
            params,
        ).fetchall()
    return [r["u"] for r in rows]


def _body_match_uuids(conn, rx, min_messages=3):
    """Conversation uuids where at least `min_messages` message bodies match `rx`.
    The threshold enforces *substantial* presence -- a couple of incidental mentions
    buried in an off-topic conversation are left to search/MCP, not auto-tagged."""
    rows = conn.execute(
        """
        SELECT conv_uuid AS u
        FROM messages
        WHERE text ~* %(rx)s
        GROUP BY conv_uuid
        HAVING count(*) >= %(min)s
        """,
        {
            "rx": rx,
            "min": min_messages,
        },
    ).fetchall()
    return [r["u"] for r in rows]


def topic_date_split_pass(
    routing=None,
    min_messages=2,
    confidence=0.78,
    verbose=True,
):
    """Body-scan a vocabulary shared by sibling categories and route each match by date.

    `routing` defaults to config.TOPIC_ROUTING (config.yaml `topic_routing`); the pass
    is a no-op when it is unset. Keys: source + seed_group name the seed list to scan
    (source.seeds[seed_group]); a conversation with >= min_messages matching messages
    is tagged `mid` when its created_at falls in mid's window, elif in early's window
    -> `early`, elif on/after `late_start` -> `late` (work that postdates both
    windows), else skipped. Skips conversations already tagged (a keyword hit is the
    stronger signal). Idempotent for method='topic'."""
    routing = config.TOPIC_ROUTING if routing is None else routing
    if not routing:
        return {}
    source_slug = routing.get("source")
    seed_group = routing.get("seed_group")
    mid_slug = routing.get("mid")
    early_slug = routing.get("early")
    late_slug = routing.get("late")
    late_start = routing.get("late_start")
    terms = (config.CATEGORIES_BY_SLUG.get(source_slug, {}).get("seeds") or {}).get(seed_group) or []
    rx = _alt_regex(terms)
    if rx is None:
        return {}
    mid = config.CATEGORIES_BY_SLUG.get(mid_slug) or {}
    early = config.CATEGORIES_BY_SLUG.get(early_slug) or {}
    mstart = _date.fromisoformat(mid["date_start"]) if mid.get("date_start") else None
    mend = _date.fromisoformat(mid["date_end"]) if mid.get("date_end") else None
    pstart = _date.fromisoformat(early["date_start"]) if early.get("date_start") else None
    pend = _date.fromisoformat(early["date_end"]) if early.get("date_end") else None
    lstart = _date.fromisoformat(late_start) if late_start else None
    data = categories.load()
    categories.clear_unlocked(data, {"topic"})
    counts = {}
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                m.conv_uuid,
                c.created_at
            FROM messages m
            JOIN conversations c ON c.uuid = m.conv_uuid
            WHERE m.text ~* %(rx)s
            GROUP BY m.conv_uuid, c.created_at
            HAVING count(*) >= %(min)s
            """,
            {
                "rx": rx,
                "min": min_messages,
            },
        ).fetchall()
        for r in rows:
            existing = set(categories.tags_for(data, r["conv_uuid"]))
            if existing:  # a title-based tag is the stronger signal -- don't override
                counts["skipped(already tagged)"] = counts.get("skipped(already tagged)", 0) + 1
                continue
            dd = r["created_at"].date() if r["created_at"] else None
            if dd and mstart and mend and mstart <= dd <= mend:
                slug = mid_slug
            elif dd and pstart and pend and pstart <= dd <= pend:
                slug = early_slug
            elif dd and lstart and dd >= lstart:  # postdates both windows -> late
                slug = late_slug
            else:  # before the late cutoff and outside both windows -- don't tag
                counts["skipped(outside windows)"] = counts.get("skipped(outside windows)", 0) + 1
                continue
            categories.apply_proposal(data, r["conv_uuid"], slug, confidence, "topic")
            counts[slug] = counts.get(slug, 0) + 1
        categories.save(data)
        categories.sync_to_db(conn, data)
    return counts


def keyword_pass(scope="meta", confidence=0.9, purpose_confidence=0.7, body_confidence=0.75, verbose=True):
    """Run the keyword layer over every configured category. Idempotent: clears prior
    keyword-method tags (on unlocked conversations) before re-applying.

    Matchers per category:
      * strong seeds -- matched in title+summary (or title only, per the category's
        `scope`), hard-gated to the date window when the category is bounded. A match
        means the conversation is genuinely ABOUT the category: it tags (method
        'keyword') and seeds the semantic centroid.
      * purpose seeds (optional) -- project/purpose terms (title+summary) that tag the
        conversation by PURPOSE (method 'keyword-purpose') WITHOUT seeding the centroid
        -- e.g. the Twitter-archive coding chats are politics-by-purpose, but their
        prose is code and would muddy the politics centroid.
      * body seeds (optional) -- ultra-distinctive terms matched in message bodies.

    Returns {slug: n_conversations_tagged}.
    """
    data = categories.load()
    categories.clear_unlocked(data, {"keyword", "keyword-purpose"})
    matched = {}
    with get_conn() as conn:
        for cat in config.CATEGORIES:
            slug = cat["slug"]
            seeds = cat.get("seeds") or {}
            cat_scope = cat.get("scope", scope)
            raw_ds, raw_de = cat.get("date_start"), cat.get("date_end")
            gate_start = _date.fromisoformat(raw_ds) if (raw_ds and raw_de) else None
            gate_end = _date.fromisoformat(raw_de) if (raw_ds and raw_de) else None
            hits = set()
            rx = _alt_regex(seeds.get("strong") or [])
            if rx is not None:
                for u in _strong_match_uuids(conn, rx, cat_scope, gate_start, gate_end):
                    categories.apply_proposal(data, u, slug, confidence, "keyword")
                    hits.add(u)
            prx = _alt_regex(seeds.get("purpose") or [])
            if prx is not None:
                # Tag by PURPOSE without seeding the centroid (e.g. the Twitter archive
                # is politics): the prose is the project's code, not the purpose's.
                for u in _strong_match_uuids(conn, prx, "meta", gate_start, gate_end):
                    categories.apply_proposal(data, u, slug, purpose_confidence, "keyword-purpose")
                    hits.add(u)
            brx = _alt_regex(seeds.get("body") or [])
            if brx is not None:
                for u in _body_match_uuids(conn, brx):
                    categories.apply_proposal(data, u, slug, body_confidence, "keyword")
                    hits.add(u)
            matched[slug] = len(hits)
            if verbose:
                print(f"  {slug:16s} matched {len(hits):4d} conversations")
        categories.save(data)
        categories.sync_to_db(conn, data)
    return matched


# ---------------------------------------------------------------- layer 2: semantic


def _bounded_window(slug):
    """(start, end) dates for a hard-bounded category (both ends set), else (None, None)."""
    cat = config.CATEGORIES_BY_SLUG.get(slug, {})
    ds, de = cat.get("date_start"), cat.get("date_end")
    if ds and de:
        return _date.fromisoformat(ds), _date.fromisoformat(de)
    return None, None


def _in_window(slug, day):
    """True if `day` falls in slug's hard window; open-ended categories never gate."""
    start, end = _bounded_window(slug)
    if start is None:
        return True
    if day is None:
        return False
    return start <= day <= end


def _window_span_days(slug):
    """Width of a category's hard window in days; open-ended sorts last (least specific)."""
    start, end = _bounded_window(slug)
    if start is None:
        return 10 ** 9
    return (end - start).days


def semantic_decisions(date_split_groups=None):
    """Score every still-untagged conversation against each category centroid and
    return the single best (date-routed) assignment per conversation, BEFORE any
    similarity threshold is applied.

    Category centroids are built from the current keyword/topic/user tags (semantic
    tags from a prior run are excluded, so the layer never bootstraps on itself).
    Conversations that already carry a tag, or are user-locked, are not candidates.
    Bounded categories are hard date-gated. `date_split_groups` defaults to
    config.DATE_SPLIT_GROUPS (config.yaml `date_split_groups`) -- categories that share
    a topic but are separated by date, whose centroids are near-identical. For those
    the match score is the group's best centroid similarity, but the assigned member
    is the one whose hard window contains the conversation (narrowest wins), mirroring
    the topic pass rather than trusting the centroids to tell them apart.

    Returns (decisions, data, seed_counts), where `decisions` is a list of
    {uuid, name, date, slug, sim} sorted by descending sim."""
    if date_split_groups is None:
        date_split_groups = config.DATE_SPLIT_GROUPS
    data = categories.load()
    seeds, tagged, locked = {}, set(), set()
    for uuid, rec in data["conversations"].items():
        if rec.get("locked"):
            locked.add(uuid)
        for slug, tag in rec.get("tags", {}).items():
            method = tag.get("method")
            if method != "semantic":
                tagged.add(uuid)                          # already classified -- not a candidate
            if method in ("keyword", "topic", "user"):
                seeds.setdefault(slug, []).append(uuid)   # genuinely about it -- seeds the centroid
    seed_counts = {slug: len(uuids) for slug, uuids in seeds.items()}
    group_of = {}
    for group in date_split_groups:
        for slug in group:
            group_of[slug] = group

    with get_conn() as conn:
        rows = semantic_centroid_scores(conn, seeds, sorted(tagged | locked))

    by_conv = {}
    for r in rows:
        rec = by_conv.setdefault(
            r["uuid"],
            {
                "name": r["name"],
                "day": r["created_at"].date() if r["created_at"] else None,
                "sims": {},
            },
        )
        rec["sims"][r["slug"]] = float(r["sim"])

    decisions = []
    for uuid, rec in by_conv.items():
        eligible = {s: sim for s, sim in rec["sims"].items() if _in_window(s, rec["day"])}
        if not eligible:
            continue
        best = max(eligible, key=eligible.get)
        group = group_of.get(best)
        if group:
            in_group = {s: eligible[s] for s in group if s in eligible}
            slug = min(in_group, key=_window_span_days)   # date routing: narrowest window
            sim = max(in_group.values())                  # detection = group's best match
        else:
            slug, sim = best, eligible[best]
        decisions.append({
            "uuid": uuid,
            "name": rec["name"],
            "date": rec["day"].isoformat() if rec["day"] else None,
            "slug": slug,
            "sim": round(sim, 4),
        })
    decisions.sort(key=lambda d: -d["sim"])
    return decisions, data, seed_counts


def semantic_pass(threshold=None, dry_run=False, verbose=True):
    """Layer 2 of the sieve. Propose a category for each untagged conversation whose
    centroid is close enough to a category centroid (see semantic_decisions).
    `threshold` defaults to config.SEMANTIC_THRESHOLD; a per-category
    `semantic_threshold` in config.yaml overrides it. Proposals are method='semantic'
    with confidence = cosine similarity. dry_run computes and reports without writing.

    Returns {decisions, proposals, counts, seed_counts, threshold, overrides, applied}."""
    if threshold is None:
        threshold = config.SEMANTIC_THRESHOLD
    overrides = {
        c["slug"]: float(c["semantic_threshold"])
        for c in config.CATEGORIES
        if c.get("semantic_threshold") is not None
    }

    def bar(slug):
        return overrides.get(slug, threshold)

    decisions, data, seed_counts = semantic_decisions()
    proposals = [d for d in decisions if d["sim"] >= bar(d["slug"])]
    counts = {}
    for d in proposals:
        counts[d["slug"]] = counts.get(d["slug"], 0) + 1

    if not dry_run:
        categories.clear_unlocked(data, {"semantic"})
        for d in proposals:
            categories.apply_proposal(data, d["uuid"], d["slug"], d["sim"], "semantic")
        categories.save(data)
        with get_conn() as conn:
            categories.sync_to_db(conn, data)

    if verbose:
        tag = " + per-category overrides" if overrides else ""
        print(f"semantic pass (threshold {threshold}{tag}):")
        for slug in config.CATEGORY_SLUGS:
            print(
                f"  {slug:16s} seeds {seed_counts.get(slug, 0):4d}"
                f"  ->  proposals {counts.get(slug, 0):4d}"
            )
        note = "  (dry run -- nothing written)" if dry_run else ""
        print(f"  {'TOTAL':16s} proposals {len(proposals):4d}{note}")

    return {
        "decisions": decisions,
        "proposals": proposals,
        "counts": counts,
        "seed_counts": seed_counts,
        "threshold": threshold,
        "overrides": overrides,
        "applied": not dry_run,
    }


def predictions_for(uuids, floor=0.2):
    """Per-conversation cosine similarity to every category centroid, date-gated and
    kept only when >= floor. Centroids are built from the CURRENT keyword/topic/user
    seeds, so this reflects the user's confirmations live. Returns {uuid: {slug: sim}}
    (each conversation's predictions sorted high to low) -- the review UI shows these so
    every plausible category is visible, not just the committed top pick."""
    targets = list(uuids)
    if not targets:
        return {}
    data = categories.load()
    seeds = {}
    for u, rec in data["conversations"].items():
        for slug, tag in rec.get("tags", {}).items():
            if tag.get("method") in ("keyword", "topic", "user"):
                seeds.setdefault(slug, []).append(u)
    with get_conn() as conn:
        rows = centroid_scores_for(conn, seeds, targets)
    out = {}
    for r in rows:
        sim = float(r["sim"])
        if sim < floor:
            continue
        day = r["created_at"].date() if r["created_at"] else None
        if not _in_window(r["slug"], day):
            continue
        out.setdefault(r["uuid"], {})[r["slug"]] = round(sim, 4)
    return {
        uuid: dict(sorted(preds.items(), key=lambda kv: -kv[1]))
        for uuid, preds in out.items()
    }
