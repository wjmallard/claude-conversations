"""Render a message's content blocks to HTML for the detail view.

Works from the raw `content` array (messages.raw), not the flattened `text` field.
Assistant/human prose is rendered as Markdown; thinking, tool calls, and tool
results become collapsible sections. Images are not rendered (by design).

Safety: prose from you/Claude is rendered as Markdown (trusted, local, single
user). Untrusted web content inside tool results is HTML-escaped and shown as
plain text -- never rendered as Markdown/HTML.
"""

import json
import re

import markdown as _markdown
from markupsafe import Markup, escape

_MD_EXT = ["fenced_code", "tables", "sane_lists", "nl2br"]

# Every summary the export writes opens with its own bold "**Conversation overview**"
# line, which only repeats the label of the panel it sits in.
_SUMMARY_HEADER_RE = re.compile(r"\A\s*\*\*conversation overview\*\*\s*\n+", re.IGNORECASE)


def _md(text) -> Markup:
    return Markup(_markdown.markdown(text or "", extensions=_MD_EXT, output_format="html5"))


def render_summary(text) -> Markup:
    """Render a conversation's summary (Markdown, like everything else the export
    writes), minus the redundant header it opens with."""
    return _md(_SUMMARY_HEADER_RE.sub("", text or "", count=1))


def _details(cls, summary_html, body_html) -> Markup:
    return (Markup('<details class="blk {}"><summary>').format(cls)
            + summary_html
            + Markup('</summary><div class="blk-body">')
            + body_html
            + Markup("</div></details>"))


def _render_tool_use(b) -> Markup:
    name = b.get("name") or "tool"
    msg = b.get("message") or ""
    inp = b.get("input")
    pretty = json.dumps(inp, indent=2, ensure_ascii=False) if inp is not None else ""
    summary = Markup('🔧 <span class="tool-name">{}</span>').format(name)
    if msg:
        summary += Markup(' <span class="muted">&mdash; {}</span>').format(msg)
    body = Markup('<pre class="json">{}</pre>').format(pretty) if pretty else Markup("")
    return _details("tool", summary, body)


def _render_tool_result(b) -> Markup:
    name = b.get("name") or "result"
    content = b.get("content")
    items = content if isinstance(content, list) else [content]
    parts = []
    for it in items:
        if isinstance(it, dict):
            if it.get("type") == "knowledge" or it.get("url"):
                title = it.get("title") or it.get("url") or "link"
                url = it.get("url") or ""
                meta = it.get("metadata") or {}
                site = meta.get("site_name") or meta.get("site_domain") or ""
                line = Markup('<div class="kn"><a href="{}" target="_blank" rel="noopener">{}</a>').format(url, title)
                if site:
                    line += Markup(' <span class="muted">{}</span>').format(site)
                parts.append(line + Markup("</div>"))
            elif it.get("text") is not None:
                parts.append(Markup('<pre class="result-text">{}</pre>').format(it["text"]))
            else:
                parts.append(Markup('<pre class="json">{}</pre>').format(
                    json.dumps(it, indent=2, ensure_ascii=False)[:4000]))
        elif isinstance(it, str):
            parts.append(Markup('<pre class="result-text">{}</pre>').format(it))
    body = Markup("").join(parts) if parts else Markup('<span class="muted">(empty)</span>')
    summary = Markup('📄 <span class="tool-name">{}</span>').format(name)
    return _details("result", summary, body)


def render_blocks(content) -> Markup:
    out = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append(_md(b.get("text", "")))
        elif t == "thinking":
            out.append(_details("thinking", Markup("🧠 thinking"), _md(b.get("text", ""))))
        elif t == "tool_use":
            out.append(_render_tool_use(b))
        elif t == "tool_result":
            out.append(_render_tool_result(b))
        elif t in ("token_budget", "image"):
            continue  # no display value / images out of scope
        else:
            out.append(_details(
                "other", escape(str(t)),
                Markup('<pre class="json">{}</pre>').format(
                    json.dumps(b, indent=2, ensure_ascii=False)[:4000]),
            ))
    return Markup("").join(out)


_ATTACH_DISPLAY_CAP = 20000


def _human_size(n) -> str:
    """Bytes -> a short human string (e.g. '48 KB'); '' if not a number."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TB"


def _render_attachments(msg) -> Markup:
    """Collapsible blocks for pasted/uploaded documents (attachments carrying
    extracted text), shown inside the turn they belong to and collapsed by default.
    Very long documents are capped for display (the full text is in the raw view)."""
    out = []
    for a in msg.get("attachments") or []:
        if not isinstance(a, dict):
            continue
        content = a.get("extracted_content") or ""
        if not content:
            continue
        name = a.get("file_name") or "document"
        size = _human_size(a.get("file_size") or len(content))
        summary = Markup('📎 <span class="att-name">{}</span>').format(name)
        if size:
            summary += Markup(' <span class="muted">· {}</span>').format(size)
        body = Markup('<pre class="att-body">{}</pre>').format(content[:_ATTACH_DISPLAY_CAP])
        if len(content) > _ATTACH_DISPLAY_CAP:
            body += Markup(
                '<div class="muted att-more">&hellip; {} more characters &mdash;'
                ' open <b>raw</b> (top of page) for the full document</div>'
            ).format(f"{len(content) - _ATTACH_DISPLAY_CAP:,}")
        out.append(_details("attachment", summary, body))
    return Markup("").join(out)


def _file_names(msg) -> list[str]:
    """Names of content-free uploads (the `files` list -- images/binaries); their
    bytes aren't in the export, so they surface only as a filename footer."""
    names = []
    for a in msg.get("files") or []:
        if isinstance(a, dict):
            names.append(a.get("file_name") or a.get("name") or a.get("title") or "file")
        elif isinstance(a, str):
            names.append(a)
    return names


def render_message(msg) -> dict:
    """Return a view model: {uuid, sender, created_at, html, files}."""
    html = render_blocks(msg.get("content"))
    if not html and msg.get("text"):
        html = _md(str(msg["text"]))
    html += _render_attachments(msg)
    return {
        "uuid": msg.get("uuid"),
        "sender": msg.get("sender") or "?",
        "created_at": msg.get("created_at"),
        "html": html,
        "files": _file_names(msg),
    }
