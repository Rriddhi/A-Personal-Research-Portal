"""
Citation explorer: click a citation to see chunk text + document metadata.
Requires run to have citation_mapping and retrieved_chunks (with optional full text).
"""
import streamlit as st


def _chunk_by_id(run: dict) -> dict:
    """Build chunk_id -> chunk dict from run's retrieved_chunks."""
    chunks = run.get("retrieved_chunks") or []
    return {c.get("chunk_id", ""): c for c in chunks if c.get("chunk_id")}


def _get_meta(source_id: str):
    """Lazy import to avoid loading manifest in UI until needed."""
    try:
        from prp.citations import get_source_meta
        return get_source_meta(source_id or "")
    except Exception:
        return {}


def render_citation_explorer(run: dict, key_suffix: str = "") -> None:
    """
    UI with:
      - list of citations (chips or selectbox)
      - on select: show chunk text (full), metadata card (title/authors/year/venue), and link/doi if present
    key_suffix: use for unique Streamlit keys when rendering multiple explorers (e.g. per run in History).
    """
    if not run:
        st.caption("No run selected.")
        return

    mapping = run.get("citation_mapping") or []
    chunk_by_id = _chunk_by_id(run)

    # Prefer citation_mapping so order matches answer; fallback to unique chunks
    if mapping:
        options = []
        seen = set()
        for m in mapping:
            cid = m.get("chunk_id") or ""
            sid = m.get("source_id") or ""
            apa = m.get("apa") or ""
            if cid and cid not in seen:
                seen.add(cid)
                label = f"{apa} — {cid}" if apa else cid
                options.append((cid, label))
    else:
        options = []
        for cid, c in chunk_by_id.items():
            sid = c.get("source_id") or ""
            meta = _get_meta(sid)
            title = (meta.get("title") or "")[:50] or sid
            options.append((cid, f"{title} — {cid}"))

    if not options:
        st.caption("No citations in this run.")
        return

    labels = [o[1] for o in options]
    chunk_ids = [o[0] for o in options]
    sel_key = f"citation_explorer_select_{key_suffix}" if key_suffix else "citation_explorer_select"
    idx = st.selectbox(
        "Select citation",
        range(len(labels)),
        format_func=lambda i: labels[i],
        key=sel_key,
    )
    if idx is None:
        return

    cid = chunk_ids[idx]
    chunk = chunk_by_id.get(cid, {})
    source_id = chunk.get("source_id") or ""
    if not source_id and mapping:
        for m in mapping:
            if m.get("chunk_id") == cid:
                source_id = m.get("source_id") or ""
                break
    meta = _get_meta(source_id)

    # Chunk text
    st.markdown("**Chunk text**")
    full_text = chunk.get("text") or chunk.get("text_preview") or ""
    if full_text:
        ta_key = f"citation_chunk_text_{key_suffix}" if key_suffix else "citation_chunk_text"
        st.text_area("", value=full_text, height=200, disabled=True, key=ta_key)
    else:
        st.info("Chunk text not stored for this run. Re-run retrieval to see full text here.")

    # Metadata card
    st.markdown("**Document metadata**")
    def _cell(s: str, max_len: int = 200) -> str:
        s = (s or "").strip() or "—"
        s = s.replace("|", " ").replace("\n", " ")[:max_len]
        return s

    title = _cell(meta.get("title"), 200)
    authors = _cell(meta.get("authors"), 200)
    year = _cell(str(meta.get("year") or ""), 20)
    venue = _cell(meta.get("venue"), 150)
    link_doi = _cell(meta.get("link_or_doi"), 150)

    st.markdown(f"""
    | Field | Value |
    |-------|-------|
    | Title | {title} |
    | Authors | {authors} |
    | Year | {year} |
    | Venue | {venue} |
    | Link / DOI | {link_doi} |
    """)
    if link_doi and link_doi != "—" and (link_doi.startswith("http") or "doi" in link_doi.lower()):
        url = link_doi if link_doi.startswith("http") else f"https://doi.org/{link_doi.lstrip('doi:').strip()}"
        st.markdown(f"[Open link]({url})")
