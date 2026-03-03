# Phase 3 Requirements (Personal Research Portal)

**Status:** No official Phase 3 specification was found in the repository. This document captures the default, recommended Phase 3 scope so the implementation remains modular and can be swapped for an official spec later.

---

## TODO: Official spec

- [ ] If an official Phase 3 assignment or PDF exists, replace the sections below with the exact requirements and acceptance criteria from that document.
- [ ] Map any official deliverables to the implemented features (sessions, export, quality gates) and add/remove as needed.

---

## Adopted scope (default implementation)

### 1. Session-based querying (persistent history)

- **Requirement:** A user can start a session, run queries within it, and have all inputs/outputs persisted.
- **Session record:** Each session contains:
  - `session_id`
  - `timestamp` (created)
  - `query_text`
  - `answer` (full Answer + Evidence + References text)
  - `citations` (list of (source_id, chunk_id))
  - `citation_mapping` (APA → chunk_id)
  - `retrieved_chunks` preview (chunk_id, source_id, text_preview)
  - `retrieval_method`
  - `query_id`
  - Optional config snapshot (e.g. top_k, model_name) for reproducibility.
- **Storage:** `logs/sessions/<session_id>.jsonl` — one JSON object per query run appended to the session file.
- **CLI:** `python -m prp session new | list | show <id> | query <id> <query_text>`
  - `new` — create a new session, print session_id.
  - `list` — list session_ids with timestamps.
  - `show <id>` — print session metadata and query history (no full chunk text by default).
  - `query <id> <query_text>` — run query in session, append result to session file, print answer.
- **Acceptance:** Session creation is deterministic (e.g. session_id from timestamp or UUID); append and reload produce consistent state; no Phase 2 behavior changed.

### 2. Exportable artifacts

- **Evidence table (CSV):**
  - Columns: `query_id`, `source_id`, `chunk_id`, `apa_citation`, `evidence_snippet`, `chunk_text_preview`
  - One row per cited chunk (or per retrieved chunk used in evidence). All cited chunk_ids must appear in the export and align with manifest/docstore.
- **Full run export (JSON):**
  - Single JSON file containing the full answer package: answer, citations, citation_mapping, retrieved_chunks (with text preview), metadata (query_id, timestamp, retrieval_method, etc.).
- **CLI:** `python -m prp export --session <session_id> --format csv|json --out <path>`
  - Export can be from a session (all queries in session) or from a single run (e.g. from logs/runs); for Phase 3, session-based export is required.
- **Acceptance:** CSV rows match citation_mapping and retrieved_chunks; no citation in the answer refers to a chunk_id missing from the export; tests assert citation alignment.

### 3. Phase 3 quality gates

- **Answer package schema:** A valid “answer package” must include:
  - `answer` (str)
  - `citations` (list)
  - `citation_mapping` (list of {apa, source_id, chunk_id})
  - `retrieved_chunks` (list with at least chunk_id, source_id, text or text_preview)
  - `metadata` (dict with at least query_id, timestamp, retrieval_method or equivalent)
- **Citation validation:** Every citation (each (source_id, chunk_id) or each entry in citation_mapping) must correspond to a `chunk_id` present in `retrieved_chunks`. Tests must fail if any citation does not match a retrieved chunk_id.
- **Acceptance:** Schema validation runs on every session append and export; unit tests enforce “no citation without retrieved chunk”.

### 4. Optional: minimal web UI

- **If feasible without heavy deps:** Minimal Streamlit UI or FastAPI + simple HTML.
- **UI must show:** Query input, answer, evidence section, references section, clickable or expandable chunk previews.
- **Constraint:** Dependencies stay minimal; Phase 2 and Phase 3 CLI remain the primary interface.

---

## Non-goals (Phase 3 default)

- Changing Phase 2 retrieval/generation logic (unless an official spec requires it).
- Adding hallucinated or non-retrieved citations.
- Replacing Phase 2 evaluation metrics or run logs; Phase 3 is additive.

---

## Deliverables checklist

- [x] docs/phase3_requirements.md (this file)
- [ ] docs/phase3_design.md
- [ ] Session module + CLI (session new | list | show | query)
- [ ] Export module + CLI (export --session --format --out)
- [ ] Quality gate: answer package schema + citation-in-retrieved validation + tests
- [ ] New tests for Phase 3
- [ ] Makefile target `phase3_demo`
- [ ] Short summary: what changed, how to run, how to test
