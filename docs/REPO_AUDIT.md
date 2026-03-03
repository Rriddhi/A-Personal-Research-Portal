# Personal Research Portal (PRP) — Full Repo Audit

**Audit date:** 2026-02-24  
**Scope:** Phase 2 codebase; preparation for Phase 3. No Phase 3 spec found in repo.

---

## A) Repo map: directories and key files

### Directory layout

```
Personal_Research_Portal/
├── prp/                    # Core package
├── scripts/                 # CLI entry scripts
├── tests/                   # Pytest tests
├── data/                    # Raw PDFs, processed outputs, manifest, eval queries
├── indexes/                 # FAISS, docstore, BM25 (built at runtime)
├── logs/                    # runs/, sessions (Phase 3)
├── metrics/                 # Evaluation outputs
├── report/                  # Phase 2 evaluation report
├── docs/                    # This audit; Phase 3 docs go here
├── Makefile
├── requirements.txt
└── README.md
```

### Component roles

| File / path | Role |
|-------------|------|
| **prp/config.py** | Central config: paths (DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEXES_DIR, LOGS_DIR, METRICS_DIR), chunk/retrieval/generation constants (CHUNK_SIZE, TOP_K, RRF_K, MAX_PER_SOURCE, BIBLIOGRAPHY_THRESHOLD, RELEVANCE_*, ONCOLOGY_TERMS, etc.), `ensure_dirs()`. Loads `.env` via python-dotenv if present. |
| **prp/ingest.py** | PDF discovery and text extraction. Reads from `data/raw/` (or optional source dir). Uses PyMuPDF word-based extraction, `normalize_extracted_text`, writes `sources.jsonl` and `manifest.csv`. Handles ZIPs (skips __MACOSX etc.). |
| **prp/chunk.py** | Section-aware chunking with fixed size + overlap. Reads `sources.jsonl`, applies `filter_acknowledgements_and_boilerplate` before chunking, writes `chunks.jsonl`. Chunk schema: chunk_id, source_id, text, start_char, end_char. |
| **prp/embed.py** | Embeddings via sentence-transformers (all-MiniLM-L6-v2). `get_embedding_model()`, `embed_texts(texts)`. No API key; local only. |
| **prp/index.py** | Builds FAISS index + docstore (from chunks) and BM25 index. `load_chunks()`, `build_faiss_index()`, `build_bm25_index()`, `build_all_indexes()`. Writes to `indexes/faiss.index`, `indexes/docstore.json`, `indexes/bm25/`. |
| **prp/retrieve.py** | Hybrid retrieval: BM25 + FAISS, RRF merge, bibliography/boilerplate filter, source diversification, relevance re-rank, topic-mismatch guardrail (oncology). `retrieve_hybrid`, `retrieve_two_pass`, `retrieve_hybrid_multi`. Loads indexes on first use. |
| **prp/generate.py** | Answer generation from retrieved chunks: LLM or extractive, trust behavior (refuse unsupported strong claims, flag conflicts, “not enough evidence”). Builds Answer / Evidence / References; internal citations `(chunk_id)` then mapped to APA. Citation validation: only cite retrieved chunk_ids. |
| **prp/citations.py** | APA formatting. Loads manifest/sources into SOURCE_MAP. `format_in_text_apa`, `format_reference_apa`, `build_internal_to_apa_map`, `build_citation_mapping`, `references_apa_sorted`. |
| **prp/evaluate.py** | Phase 2 evaluation: context precision/recall, faithfulness, citation precision, answer relevance, coherence, conciseness, artifact readiness. Reads `eval_queries.json`, uses `run_query`, writes metrics to `metrics/`. |
| **prp/run.py** | Pipeline orchestration: `run_query()` (retrieve + generate, append to phase2_runs.jsonl, save per-run JSON under logs/runs/), `run_demo()` (ensure ingest+chunk+index then one query). Off-topic chunk filter (oncology) and enhancement mode. |
| **prp/query_decompose.py** | Lightweight compare-query detection and subquery building (no LLM). `is_compare_query`, `build_subqueries` for multi-query retrieval. |
| **prp/utils.py** | Logging, `source_id_from_path`, `append_run_log`, `save_run_to_file`, `utc_timestamp`, text normalization (`fix_concatenated_words`, `normalize_extracted_text`), boilerplate/bibliography scoring and filters (`bibliography_score`, `is_bibliography_chunk`, `is_boilerplate_with_bib_threshold`, `filter_acknowledgements_and_boilerplate`), evidence-type labels, `preview_from_clean`. |
| **prp/__main__.py** | Entry for `python -m prp`: commands `run_demo`, `query`, `eval`, `eval_full`. |

### scripts/

| Script | Role |
|--------|------|
| **run_ingest.py** | Ingest CLI. Default source `data/raw/`; `--source` for alternate (e.g. Research_Papers). Calls `ingest_corpus()`. |
| **build_index.py** | One-shot: `ensure_dirs` → `ingest_corpus` → `run_chunking` → `build_all_indexes`. |
| **run_query.py** | Single query: `run_query(query, query_id)`, then `render_response()` (Answer block; optional `--debug` for retrieval/citations). |
| **run_eval.py** | Runs `run_evaluation()`, `write_phase2_metrics()`, `write_evaluation_summary_md()`. Writes metrics/*. |
| **run_evaluation_full.py** | Full evaluation script (invoked by `python -m prp eval_full`). |
| **run_stress_tests.py** | Stress tests; writes `logs/stress_tests_v3.txt`. |
| **merge_bib_manifest.py** | Merge Zotero BibTeX into manifest; `--write` overwrites manifest.csv. |
| **enrich_manifest.py** | Enrich manifest metadata. |
| **check_chunks_ack_filter.py** | Check ack/funding filter impact on chunks. |
| **check_duplicate_pdfs.py** | Duplicate PDF detection. |
| **test_*.py** (in scripts/) | One-off checks (extraction normalization, citation format, retrieval diversification, evidence mix, privacy, no boilerplate citations). |

### tests/

| File | Role |
|------|------|
| **test_filters_and_evidence.py** | Bibliography filter, ack/funding filter, retrieval returns non-boilerplate chunks, evidence-type classifier. |
| **test_output_format.py** | Answer has no citations; Evidence citation format; Supports: format; References include cited sources; strip_all_citations. |
| **test_citations_apa.py** | APA in-text and reference formatting. |
| **test_evidence_mix_bermingham.py** | Bermingham (Nature Medicine RCT) counted as RCT; source in docstore. |

---

## B) External artifacts used at runtime

| Artifact | Purpose |
|----------|---------|
| **data/raw/** | Raw PDFs (and optionally other ingest sources). Required for ingest. |
| **data/processed/sources.jsonl** | One JSON object per source (source_id, text, metadata). Produced by ingest; consumed by chunk + index. |
| **data/processed/chunks.jsonl** | One JSON per chunk (chunk_id, source_id, text, start_char, end_char). Produced by chunk; consumed by index + generate/citations. |
| **data/manifest.csv** | Source metadata (source_id, title, authors, year, type, venue, link_or_doi, etc.). Used by citations and evaluation. |
| **data/eval_queries.json** | 20 eval queries (query_id, query_text, category). Used by run_eval. |
| **indexes/faiss.index** | FAISS vector index. Built from chunk embeddings. |
| **indexes/docstore.json** | List of {chunk_id, source_id, text}. Used by retrieve and generation. |
| **indexes/bm25/** | meta.json (chunk_ids), corpus.json (tokenized), bm25.pkl. BM25 retrieval. |
| **logs/runs/** | Per-run JSON files (timestamped) and **phase2_runs.jsonl** (append-only log). Written by run_query / save_run_to_file / append_run_log. |
| **metrics/** | aggregate_metrics.json, per_query_metrics.json, evaluation_summary.md. Written by run_eval. |

---

## C) CLI entry points

### Makefile targets

| Target | Command | Purpose |
|--------|---------|---------|
| **ingest** | `python scripts/run_ingest.py` | Ingest PDFs → sources.jsonl, manifest.csv |
| **merge_bib_manifest** | `python scripts/merge_bib_manifest.py` | Preview BibTeX merge (writes manifest_from_bib.csv) |
| **merge_bib_manifest_write** | `python scripts/merge_bib_manifest.py --write` | Overwrite manifest.csv with BibTeX merge |
| **build_index** | `python scripts/build_index.py` | Ingest → chunk → FAISS + BM25 |
| **phase2_demo** | `python -m prp run_demo` | Build if needed, run one query, print answer + citations + mapping |
| **run_eval** | `python scripts/run_eval.py` | Run 20-query evaluation, write metrics/ |
| **enrich_manifest** | `python scripts/enrich_manifest.py` | Enrich manifest |
| **clean** | `rm -rf data/processed indexes metrics` | Remove processed data and indexes (keeps data/raw) |
| **run_query** | Echo usage | `python scripts/run_query.py "query" [query_id]` |
| **test** | `python -m pytest tests/test_filters_and_evidence.py -v` | Run filter/evidence tests |

### python -m prp

| Command | Behavior |
|---------|----------|
| `run_demo` | run_demo(): build pipeline if missing, one query, print output, append log |
| `query <query_text>` | run_query(query_text), print answer |
| `eval` | run_evaluation(), write Phase 2 metrics and summary MD |
| `eval_full` | Subprocess to scripts/run_evaluation_full.py |

### scripts/run_*.py

- **run_query.py** — `python scripts/run_query.py "Your question" [query_id]`; `--debug` for extra fields.
- **run_eval.py** — no args; uses data/eval_queries.json and writes metrics/.
- **run_ingest.py** — optional `--source <dir>`.
- **build_index.py** — no args.

---

## D) Guardrails and quality filters

### Bibliography filter (retrieval)

- **Config:** `BIBLIOGRAPHY_THRESHOLD = 0.5`, `BIBLIOGRAPHY_THRESHOLD_RELAXED = 0.7` (config.py).
- **Logic:** `utils.bibliography_score(text)` (headings, citation patterns, semicolon density, author-year, numbered refs) → chunk excluded if score ≥ threshold. In `retrieve._merged_chunks_from_rrf_scores`, candidates filtered with `is_boilerplate_with_bib_threshold(..., bibliography_threshold)`. If fewer than top_k after strict threshold, retry with relaxed (0.7).
- **Location:** prp/retrieve.py (filter before diversification); prp/utils.py (bibliography_score, is_bibliography_chunk, is_boilerplate_with_bib_threshold).

### Boilerplate / acknowledgements filtering

- **Pre-chunk (chunk.py):** `filter_acknowledgements_and_boilerplate(text)` removes lines matching ACK_FUNDING_PATTERNS and similar (supported by, funded by, grant, acknowledg, Received:, etc.) and boilerplate line patterns (utils._is_boilerplate_line).
- **Retrieval (retrieve.py):** Same logic via `is_boilerplate_with_bib_threshold` (ack/funding, methodology boilerplate, reference list, figure/table-only, plus bibliography threshold).
- **Generation (generate.py):** Skips bibliography chunks when picking evidence quotes; filters evidence junk (figure/table captions, etc.) and answer boilerplate (ANSWER_BOILERPLATE_RE, BANNED_NONCONTENT_SUBSTRINGS).

### Topic mismatch guardrail (oncology)

- **Config:** `TOPIC_MISMATCH_GUARDRAIL`, `ONCOLOGY_TERMS`, `TOPIC_MISMATCH_PENALTY` (config.py).
- **Logic:** In `retrieve._rerank_by_relevance`: if query does not mention oncology terms, chunks containing them are downweighted (topic-mismatch penalty). In `run._filter_off_topic_chunks`: chunks with oncology terms can be dropped unless query mentions them (soft: keep all if filtering would leave &lt; top_k/2).

### Citation validation

- **generate.py:** `validate_and_clean_citations`: parse internal citations `(chunk_id)`, restrict to `valid_chunk_ids` (from retrieved_chunks); remove invalid, log warning. Only chunk_ids from retrieved chunks are mapped to APA; no hallucinated citations.
- **evaluate.py:** `compute_citation_precision`: citations (from Evidence or citations list) must resolve to `retrieved_chunks` chunk_ids; precision = valid / total.

---

## E) Run/verify

| Action | Result |
|--------|--------|
| **make test** | **PASSED** — `pytest tests/test_filters_and_evidence.py -v` (4 tests). |
| **All tests** | **PASSED** — `pytest tests/` (13 tests: filters, evidence, output format, citations APA, evidence mix Bermingham). |
| **make phase2_demo** | **FAILED** in this environment — 403 Forbidden when downloading sentence-transformers model (network/proxy). Indexes and data exist; failure is environmental, not code. Pipeline flow (ingest → chunk → index → retrieve → generate) is unchanged. |
| **make run_eval** | Not run (same embedding download would be required). |

**Conclusion:** Phase 2 tests pass. Phase 2 demo fails only due to network restriction when loading the embedding model; no code change required for the audit. For Phase 3, keep all Phase 2 behaviors and add only additive paths (new commands, new files, new targets).

---

## Phase 3 spec status

- **Searched:** README, report/, repo for “Phase 3”, “phase3”.
- **Result:** No Phase 3 requirements document found in the repository.
- **Next:** Add `docs/phase3_requirements.md` (with TODOs if official spec is external) and `docs/phase3_design.md` before implementing a default Phase 3 feature set (sessions, export, quality gates) that remains modular for later substitution of an official spec.

---

*End of audit.*
