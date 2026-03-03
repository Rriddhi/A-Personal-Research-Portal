# Phase 3 Upgrades

Summary of Phase 3 compliance additions: PDF export, thread/session export, citation explorer, and UI polish.

## PDF Export

- **Location:** `prp/pdf_export.py`
- **Functions:**
  - `evidence_table_to_pdf(evidence_rows, title, subtitle=None)` — builds a PDF table from evidence rows (Claim, Evidence snippet, Citation, Source, Page). Used in the **Artifacts** tab for the evidence table.
  - `memo_to_pdf(memo_md, title, metadata=None)` — turns a synthesis memo (markdown-style) into a PDF with simple headings and paragraphs. Used in **Artifacts** for the synthesis memo.
  - `run_to_pdf(run, title)` — one PDF per run with question, answer, citations list, and top retrieved chunks. Used in **History** (per-run “Download Run PDF”) and inside session ZIP.
- **UI:** Artifacts tab has “Download PDF” next to MD/CSV for the evidence table and “Download memo PDF” for the synthesis memo. History tab offers “Download Run PDF” for each run.
- **Dependency:** `reportlab` (added in `requirements.txt`).

## Thread / Session Export

- **Location:** `prp/thread_export.py`
- **Functions:**
  - `run_to_markdown(run)` — full run as Markdown: question, answer, citations, evidence summary, retrieved chunks, metadata.
  - `session_to_jsonl_bytes(session_path)` — returns the raw session file bytes (one JSON object per line). `session_path` can be a session ID (e.g. UUID) or a path; for an ID, `logs/sessions/<id>.jsonl` is used.
  - `session_to_zip_bytes(session_id, runs, include_pdf=True)` — ZIP containing:
    - `session.jsonl` (exact copy of the session file),
    - `runs/<run_id>.json` (each run as JSON),
    - `runs/<run_id>.md` (each run as Markdown),
    - `runs/<run_id>.pdf` (each run as PDF when `include_pdf=True`).
- **UI:** In **History**, for the selected session: “Download Session JSONL” and “Download Session ZIP”. For each run: “Download Run JSON”, “Download Run Markdown”, “Download Run PDF”.

## Citation Explorer

- **Location:** `app/components/citation_explorer.py`
- **Function:** `render_citation_explorer(run, key_suffix="")` — Streamlit UI that:
  - Shows a list of citations (from `citation_mapping` or `retrieved_chunks`) in a selectbox.
  - On selection: shows full chunk text (or “Chunk text not stored…” for older runs), and a metadata card (title, authors, year, venue, link/DOI). If a link/DOI is present, an “Open link” control is shown.
- **Data:** Runs saved *after* the change store full chunk `text` in `retrieved_chunks` (see `prp/session.py` `_build_package_for_storage`). Older runs only have `text_preview`; the explorer then shows “Chunk text not stored; re-run retrieval.”
- **UI:** Citation explorer appears in the **Ask** tab (expander “Citation explorer”) and in the **History** tab inside each run expander. Use `key_suffix` when rendering multiple explorers (e.g. per run in History) so Streamlit keys stay unique.

## Run Schema (source of truth)

- **Location:** `prp/run_schema.py`
- **Content:** TypedDict-style definitions for `RunDict`, `RunRetrievedChunk`, and `RunCitationMapping` used by session storage, export, and UI. Session JSONL remains backward-compatible; new runs may add `text` on each chunk.

## UI Polish

- **Ask tab:** Two-column layout when a run exists: left = answer, copy buttons (“Copy answer”, “Copy citations” via download), diagnostics; right = evidence panel with citations grouped by document and expanders for chunk previews. Progress: “Retrieving relevant chunks…” then “Generating answer…”.
- **Search tab:** Each hit is shown as a card (title, year, venue), short preview with query-term highlighting (`**term**`), expander for full chunk, and “Use as citation” to add the chunk to a session-state citation clipboard.
- **Evaluation tab:** Sortable per-query table (query_id, query text, faithfulness, citation_precision, relevance, coherence) with a “Sort by” dropdown and “Open as run”: choosing a row and clicking “Populate Ask tab with this query” sets the Ask tab question so the user can run that eval query again from Ask.

## Backward Compatibility

- Session JSONL lines may omit `text` on `retrieved_chunks`; the citation explorer and run PDF handle missing text with a clear fallback message.
- No existing exports or keys were removed; new options (e.g. PDF, ZIP, Run MD/JSON) were added alongside existing ones.
