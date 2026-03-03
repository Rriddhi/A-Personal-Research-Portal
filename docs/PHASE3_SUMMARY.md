# Phase 3 Summary: What Changed, How to Run, How to Test

## What changed

- **New modules (additive; Phase 2 unchanged):**
  - **prp/quality.py** — Answer package schema validation and citation alignment: every citation must correspond to a `chunk_id` in `retrieved_chunks`. Used before appending to a session and before export.
  - **prp/session.py** — Session CRUD: create (new), list, show, append query run (query). Sessions stored as `logs/sessions/<session_id>.jsonl` (one JSON line per run).
  - **prp/export.py** — Export a session to CSV (evidence table: query_id, source_id, chunk_id, apa_citation, evidence_snippet, chunk_text_preview) or JSON (full run list).
- **Config:** `SESSIONS_DIR = LOGS_DIR / "sessions"` and `ensure_dirs()` now creates `SESSIONS_DIR`.
- **CLI:** `python -m prp` supports:
  - `session new | list | show <id> | query <id> <query_text>`
  - `export --session <id> --format csv|json --out <path>`
- **Makefile:** `phase3_demo` (create session → run one query → export CSV + JSON), `test_phase3` (run Phase 3 tests).
- **Docs:** `docs/phase3_requirements.md`, `docs/phase3_design.md`, this summary. README updated with Phase 3 commands and directory layout.
- **Tests:** `tests/test_phase3.py` — 8 tests for quality (schema, citation-in-retrieved), session (create/append/reload), export (CSV columns, citation alignment, CSV/JSON round-trip).

No Phase 2 code paths were modified; `run_query`, retrieval, and generation are unchanged.

---

## How to run

1. **Phase 2 (unchanged):**  
   `make ingest`, `make build_index`, `make phase2_demo`, `make run_eval`, `python -m prp query "..."`.

2. **Phase 3 session + export:**  
   ```bash
   python -m prp session new                    # get session_id
   python -m prp session query <session_id> "What evidence exists for personalized health?"
   python -m prp session show <session_id>
   python -m prp export --session <session_id> --format csv --out evidence.csv
   python -m prp export --session <session_id> --format json --out runs.json
   ```

3. **Phase 3 one-command demo:**  
   ```bash
   make phase3_demo
   ```  
   Creates a session, runs one query in it, exports to `logs/sessions/phase3_export_sample.csv` and `phase3_export_sample.json`, and prints the session ID.

---

## How to test

- **Phase 2 (unchanged):**  
  ```bash
  make test
  # or
  python -m pytest tests/test_filters_and_evidence.py -v
  ```

- **Phase 3 only:**  
  ```bash
  make test_phase3
  # or
  python -m pytest tests/test_phase3.py -v
  ```

- **All tests (Phase 2 + Phase 3):**  
  ```bash
  python -m pytest tests/ -v
  ```  
  (21 tests total; Phase 2 behavior must remain passing.)

---

## Deliverables checklist

- [x] docs/phase3_requirements.md
- [x] docs/phase3_design.md
- [x] Session module + CLI (session new | list | show | query)
- [x] Export module + CLI (export --session --format --out)
- [x] Quality gate (answer package schema + citation-in-retrieved validation) + tests
- [x] New tests for Phase 3 (tests/test_phase3.py)
- [x] Makefile target `phase3_demo` and `test_phase3`
- [x] Short summary (this file) + README update
