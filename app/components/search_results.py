"""
Search result card renderer for the Search tab.
Renders results as scannable cards with title, metadata, snippet, and citation actions.
"""
from __future__ import annotations

import html
import re
import streamlit as st

# Resolve title/metadata/APA (same as app tab_search)
from prp.citations import get_source_meta, format_in_text_apa
from prp.generate import normalize_chunk_citation_for_display


SNIPPET_MAX_CHARS = 400

# Subtle card styling (theme-friendly: no bright colors)
_SEARCH_CARD_CSS = """
<style>
.search-result-card {
    padding: 1rem 1.1rem;
    margin-bottom: 0.5rem;
    border-radius: 6px;
    border: 1px solid rgba(128, 128, 128, 0.35);
    background-color: rgba(128, 128, 128, 0.08);
}
.search-result-card .search-card-title { font-size: 1.1rem; font-weight: 700; margin: 0 0 0.25rem 0; }
.search-result-card .search-card-meta { font-size: 0.8rem; opacity: 0.85; margin: 0 0 0.5rem 0; }
.search-result-card .search-card-snippet { font-size: 0.95rem; line-height: 1.45; margin: 0; }
</style>
"""


def _normalize_snippet_spacing(text: str) -> str:
    """Insert space before capital letters after lowercase or period to fix run‑together words."""
    if not text:
        return text
    return re.sub(r"([a-z.])([A-Z])", r"\1 \2", text)


def highlight_terms(text: str, query: str) -> str:
    """
    Simple highlight: bold query tokens that appear in text.
    Avoid regex complexity. Must be safe for markdown.
    """
    if not text or not query:
        return text
    tokens = [t for t in query.split() if len(t) > 1]
    if not tokens:
        return text
    result: list[str] = []
    i = 0
    lower = text.lower()
    while i < len(text):
        matched = False
        for token in tokens:
            if len(token) > len(text) - i:
                continue
            if lower[i : i + len(token)] == token.lower():
                result.append("**")
                result.append(text[i : i + len(token)])
                result.append("**")
                i += len(token)
                matched = True
                break
        if not matched:
            result.append(text[i])
            i += 1
    return "".join(result)


def _snippet_to_html(snippet_markdown: str) -> str:
    """Convert snippet with **bold** to HTML with <strong>; escape the rest."""
    parts = snippet_markdown.split("**")
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append("<strong>")
            out.append(html.escape(p))
            out.append("</strong>")
        else:
            out.append(html.escape(p))
    return "".join(out)


def _meta_caption(chunk: dict, method: str) -> str:
    """Build compact metadata line: year • venue • relevance if present."""
    source_id = chunk.get("source_id") or ""
    try:
        meta = get_source_meta(source_id)
        year = (meta.get("year") or "").strip()
        venue = (meta.get("venue") or meta.get("journal") or "").strip()
        parts = []
        if year:
            parts.append(year)
        else:
            parts.append("Unknown year")
        if venue:
            parts.append(venue)
        if not parts:
            parts = ["Unknown year"]
    except Exception:
        parts = ["Unknown year"]
    score = chunk.get("score")
    if score is not None:
        try:
            parts.append(f"Relevance: {float(score):.2f}")
        except (TypeError, ValueError):
            pass
    return " • ".join(parts)


def render_search_results(
    results: list[dict],
    query: str,
    *,
    key_prefix: str = "search",
    method: str = "semantic similarity",
) -> None:
    """
    Render results as clean 'cards':
      - Title (bold)
      - Metadata line (year • venue/journal • type) as caption/badges
      - Snippet preview (max 3–4 lines, ~300–450 chars)
      - Actions row: expander "View full chunk" + "Use as citation" button
      - Divider between cards
    Must be robust to missing fields.
    """
    if not results:
        return

    # Inject card CSS once (theme-friendly)
    st.markdown(_SEARCH_CARD_CSS, unsafe_allow_html=True)

    # Header summary above results
    sort_label = method if method else "semantic similarity"
    st.caption(f"Showing top **{len(results)}** results • Sorted by: {sort_label}")

    for i, c in enumerate(results):
        source_id = c.get("source_id", "")
        chunk_id = c.get("chunk_id", "")
        full_text = c.get("text", "") or c.get("preview", "")
        preview_raw = c.get("preview", "") or full_text

        # Resolve title and APA (for citation button)
        try:
            meta = get_source_meta(source_id)
            title = (meta.get("title") or "").strip()
            if not title:
                title = normalize_chunk_citation_for_display(chunk_id, source_id)
            elif len(title) > 70:
                title = title[:67] + "..."
            apa = format_in_text_apa(source_id, meta)
        except Exception:
            title = normalize_chunk_citation_for_display(chunk_id, source_id)
            apa = ""

        # Snippet: use full text when available, normalize spacing, limit length, highlight query
        raw_snippet = (preview_raw or full_text)[:SNIPPET_MAX_CHARS]
        if len(preview_raw or full_text) > SNIPPET_MAX_CHARS:
            raw_snippet += "…"
        raw_snippet = _normalize_snippet_spacing(raw_snippet)
        snippet_highlighted = highlight_terms(raw_snippet, query)
        snippet_html = _snippet_to_html(snippet_highlighted)
        caption = _meta_caption(c, method)

        # Single card block: bordered container with title, metadata, snippet
        card_html = (
            f'<div class="search-result-card">'
            f'<p class="search-card-title">{html.escape(title)}</p>'
            f'<p class="search-card-meta">{html.escape(caption)}</p>'
            f'<p class="search-card-snippet">{snippet_html}</p>'
            f"</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Primary action first, then expander
        if st.button("Use as citation", key=f"{key_prefix}_cite_{i}", type="primary"):
            clip = st.session_state.get("_search_citation_clipboard") or []
            clip.append({
                "source_id": source_id,
                "chunk_id": chunk_id,
                "title": title,
                "apa": apa,
            })
            st.session_state._search_citation_clipboard = clip
            st.success("Added to citation clipboard.")

        with st.expander("View full chunk ▾", expanded=False):
            st.text(full_text)

        st.divider()
