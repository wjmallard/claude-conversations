"""Category-tag curation store.

The curation (which conversation has which category tags) lives on disk in a
single JSON file (config.CATEGORIES_PATH) — that file is the source of truth.
The conversations.categories array column in PostgreSQL is a rebuildable index
of it, used for fast filtering/faceting in the web UI.

File shape:

    {
      "version": 1,
      "conversations": {
        "<conv-uuid>": {
          "locked": false,                 # true once the user edits it by hand
          "tags": {
            "example-topic":   {"confidence": 0.92, "method": "semantic"},
            "another-topic":   {"confidence": 1.0,  "method": "user"}
          }
        }
      }
    }

`method` records how a tag was assigned:
  * keyword — a strong seed matched the title/summary: the conversation is
    genuinely ABOUT the category. Seeds the semantic centroid.
  * keyword-purpose — a purpose seed matched: the project exists FOR the category
    (a coding chat built for, say, a political archive). Tags without seeding the
    centroid.
  * topic — a shared body vocabulary matched, routed to one of several sibling
    categories by date window. Yields to an existing tag (a title hit is stronger).
  * semantic — proposed by cosine similarity to the category centroid; surfaced in
    the review UI for confirmation.
  * user — set by hand in the web UI.

A user edit sets the exact tag set, marks the conversation `locked`, and the
classifier then leaves it alone — so manual corrections are sticky across re-runs
of the sieve.
"""

import json
import os
import tempfile

from claude_conversations import config

VERSION = 1
USER_METHOD = "user"


def valid_slugs():
    return set(config.CATEGORY_SLUGS)


def load():
    """Return the curation dict, initializing an empty structure if absent."""
    path = config.CATEGORIES_PATH
    if not path.exists():
        return {"version": VERSION, "conversations": {}}
    with open(path) as f:
        data = json.load(f)
    data.setdefault("version", VERSION)
    data.setdefault("conversations", {})
    return data


def save(data):
    """Atomically write the curation dict to disk (temp file + rename)."""
    path = config.CATEGORIES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".categories.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def tags_for(data, uuid):
    """Sorted list of category slugs assigned to a conversation."""
    return sorted((data["conversations"].get(uuid) or {}).get("tags", {}).keys())


def is_locked(data, uuid):
    return bool((data["conversations"].get(uuid) or {}).get("locked"))


def set_user_tags(data, uuid, slugs):
    """Set a conversation's tags to exactly `slugs` (a manual edit). Sticky: marks
    the conversation locked so the classifier won't change it. Mutates `data`."""
    valid = valid_slugs()
    tags = {s: {"confidence": 1.0, "method": USER_METHOD} for s in slugs if s in valid}
    data["conversations"][uuid] = {"locked": True, "tags": tags}
    return data


def apply_proposal(data, uuid, slug, confidence, method):
    """Add/refresh a classifier-proposed tag, unless the conversation is locked
    (user-edited) — then leave it untouched. Keeps the higher confidence when the
    tag already exists. Mutates `data`."""
    if slug not in valid_slugs():
        return data
    rec = data["conversations"].get(uuid)
    if rec is None:
        rec = {"locked": False, "tags": {}}
        data["conversations"][uuid] = rec
    if rec.get("locked"):
        return data
    existing = rec["tags"].get(slug)
    if existing is None or confidence >= existing.get("confidence", 0):
        rec["tags"][slug] = {"confidence": round(float(confidence), 4), "method": method}
    return data


def clear_unlocked(data, methods=None):
    """Drop tags from non-locked conversations so a classifier layer can be re-run
    idempotently. If `methods` (a set) is given, only tags assigned by those
    methods are removed; otherwise all tags on unlocked records are cleared.
    Mutates `data`."""
    for uuid in list(data["conversations"].keys()):
        rec = data["conversations"][uuid]
        if rec.get("locked"):
            continue
        if methods is None:
            rec["tags"] = {}
        else:
            rec["tags"] = {
                s: t for s, t in rec.get("tags", {}).items()
                if t.get("method") not in methods
            }
        if not rec["tags"]:
            del data["conversations"][uuid]
    return data


def sync_to_db(conn, data=None):
    """Mirror the curation file into conversations.categories. The file is
    authoritative: each conversation's column is set to its file tags (or '{}')."""
    if data is None:
        data = load()
    convs = data.get("conversations", {})
    conn.execute("UPDATE conversations SET categories = '{}'::text[] WHERE categories <> '{}'::text[]")
    rows = [
        {"uuid": uuid, "tags": sorted(rec.get("tags", {}).keys())}
        for uuid, rec in convs.items()
        if rec.get("tags")
    ]
    if rows:
        conn.cursor().executemany(
            "UPDATE conversations SET categories = %(tags)s WHERE uuid = %(uuid)s",
            rows,
        )
    return len(rows)


def method_map(data=None):
    """{uuid: {slug: method}} over all tagged conversations — lets the UI tell
    semantic *proposals* apart from confirmed (user) and auto (keyword/topic) tags."""
    if data is None:
        data = load()
    return {
        uuid: {slug: tag.get("method") for slug, tag in rec.get("tags", {}).items()}
        for uuid, rec in data["conversations"].items()
        if rec.get("tags")
    }


def proposal_queue(data=None):
    """Conversations carrying at least one semantic proposal, for the review UI.

    Returns (queue, counts):
      queue  — [{uuid, proposed: {slug: confidence}, other: {slug: method}, score}],
               sorted by descending top-proposal confidence;
      counts — {slug: n} proposals per category, for the review filter bar.
    Locked (already-reviewed) conversations are skipped."""
    if data is None:
        data = load()
    queue, counts = [], {}
    for uuid, rec in data["conversations"].items():
        if rec.get("locked"):
            continue
        tags = rec.get("tags", {})
        proposed = {
            slug: tag.get("confidence", 0.0)
            for slug, tag in tags.items()
            if tag.get("method") == "semantic"
        }
        if not proposed:
            continue
        other = {
            slug: tag.get("method")
            for slug, tag in tags.items()
            if tag.get("method") != "semantic"
        }
        queue.append({
            "uuid": uuid,
            "proposed": proposed,
            "other": other,
            "score": max(proposed.values()),
        })
        for slug in proposed:
            counts[slug] = counts.get(slug, 0) + 1
    queue.sort(key=lambda entry: -entry["score"])
    return queue, counts
