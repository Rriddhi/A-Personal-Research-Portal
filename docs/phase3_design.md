# Phase 3 Design (Personal Research Portal)

**Goals:** Add session-based querying, exportable artifacts (CSV + JSON), and quality gates without changing Phase 2 behavior.

---

## Goals / non-goals

| Goals | Non-goals |
|-------|-----------|
| Persistent query sessions (history per session) | Changing Phase 2 retrieval/generate logic |
| Export evidence table (CSV) and full run (JSON) | New retrieval algorithms (e.g. multi-step) in Phase 3 default |
| Validate answer package schema and citation ↔ chunk alignment | Replacing phase2_runs.jsonl or eval metrics |
| Add tests for every Phase 3 requirement | Heavy UI dependencies |

---

## Architecture (text diagram)

```
                    ┌─────────────────────────────────────────┐
                    │  CLI: python -m prp                      │
                    │  session new|list|show|query              │
                    │  export --session ID --format csv|json    │
                    └─────────────────┬─────────────────────────┘
                                      │
    ┌─────────────────────────────────▼─────────────────────────┐
    │  prp/session.py          prp/export.py    prp/quality.py   │
    │  - create_session()      - export_csv()   - validate_package()
    │  - list_sessions()       - export_json()   - validate_citations()
    │  - get_session()         - from session                    │
    │  - append_query_run()    - from run dict                   │
    └───────────────────────────────────────────────────────────┘
                    │                    │
                    ▼                    ▼
    ┌───────────────────────┐  ┌──────────────────────────────┐
    │ logs/sessions/        │  │ prp/run.run_query()           │
    │ <session_id>.jsonl    │  │ (unchanged Phase 2)            │
    └──────────────────────┘  └──────────────────────────────┘
```

- **session.py** creates/reads/updates session files under `logs/sessions/`. It calls `run.run_query()` and appends the result (as an “answer package”) to the session file. No change to `run_query()` signature or return shape for Phase 2 callers.
- **export.py** reads a session file (or a single run dict), validates with **quality.py**, then writes CSV or JSON. Export uses only data from the session/run (no extra retrieval).
- **quality.py** defines the answer-package schema and validates that every citation maps to a retrieved chunk_id. Used on session append and before export.

---

## New modules

| Module | Purpose |
|--------|---------|
| **prp/session.py** | Session CRUD: create (new), list, get (show), append query run (query). Path: `LOGS_DIR / "sessions" / f"{session_id}.jsonl"`. Each line is one JSON object (one query run). |
| **prp/export.py** | Export session (or single run) to CSV (evidence table) or JSON (full run). Uses citation_mapping + retrieved_chunks; evidence_snippet derived from answer/evidence section or citations. |
| **prp/quality.py** | `validate_answer_package(pkg) -> (bool, list[str] errors)`; `validate_citations_in_retrieved(pkg) -> (bool, list[str] errors)`. Required keys: answer, citations, citation_mapping, retrieved_chunks, metadata. |

---

## Data model

- **Session file:** `logs/sessions/<session_id>.jsonl`. Each line = one run record (same shape as `run_query` return value, serializable: log_record + answer, citations, citation_mapping, retrieved_chunks with text_preview, metadata).
- **Config:** Add `SESSIONS_DIR = LOGS_DIR / "sessions"` in config.py; ensure_dirs() creates SESSIONS_DIR.
- **No DB:** No sqlite; jsonl only to keep dependencies minimal and Phase 2 untouched.

---

## Integration with run_query / retrieve / generate

- **run_query()** is unchanged. Session module calls `run_query(query_text, query_id=...)`, then builds an “answer package” from the return value (answer, citations, citation_mapping, retrieved_chunks, metadata), validates with quality.py, then appends one JSON line to the session file.
- **Export** reads the session file, parses each line, optionally validates each package, then aggregates for CSV (one row per cited chunk per query) or writes one JSON per run or one combined JSON for the session (design choice: one JSON per run for simplicity).

---

## Testing

- **Unit:**
  - Session: create session, append two runs, reload, assert line count and keys.
  - Export: from a fixture run dict, export CSV and JSON; assert CSV has expected columns and citation_mapping chunk_ids appear in CSV; assert JSON has required keys.
  - Quality: valid package passes; package missing `citation_mapping` fails; package with citation not in retrieved_chunks fails.
- **Integration:** `session new` → `session query <id> "query"` → `session show <id>` → `export --session <id> --format csv --out /tmp/out.csv`; then run tests that parse CSV and assert citation alignment.
- **Phase 2 regression:** All existing tests (test_filters_and_evidence, test_output_format, test_citations_apa, test_evidence_mix_bermingham) remain unchanged and passing.

---

## Migration / backward compatibility

- Phase 2: no code paths changed. `make test`, `make phase2_demo`, `make run_eval` behave as before.
- Phase 3: new CLI subcommands and new files only. `make phase3_demo` will: create a session, run one query in it, export to CSV and JSON, then print summary (and optionally run quality validation).

---

## Makefile

- Add target: `phase3_demo`: create session, run one query in session, export to `logs/sessions/phase3_export_sample.csv` and `.json`, print “Phase 3 demo complete. Session ID: …”.
