# Personal Research Portal — Phase 2: Research-Grade RAG

Phase 2 builds a baseline RAG pipeline over the local corpus in **`data/raw/`** (optional: `Research_Papers/`), with production patterns: logging, reproducibility, trust behavior (no invented citations; refuse unsupported strong claims; flag missing/conflicting evidence), and **hybrid retrieval** (BM25 + FAISS with reciprocal rank fusion).

## Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies (pinned for reproducibility):**
   ```bash
   pip install -r requirements.txt
   ```
   Use the **same environment** (e.g. your active conda env or venv) for `pip install` and for running commands. If you see `ModuleNotFoundError: No module named 'faiss'`, install in that env: `pip install faiss-cpu` (or re-run `pip install -r requirements.txt` there).

   **If you see** `ImportError: cannot import name 'runtime_version' from 'google.protobuf'` **(e.g. in conda envs with TensorFlow/transformers):** upgrade protobuf:  
   `pip install --upgrade 'protobuf>=4.21'`

3. **Corpus:** Place PDFs in **`data/raw/`**. Optionally use `Research_Papers/` and run `python scripts/run_ingest.py --source Research_Papers` to copy into `data/raw/`. No live links required; all processing is local.

   **Recommended run order:**
   ```bash
   make ingest
   make merge_bib_manifest_write
   make build_index
   make phase2_demo
   make run_eval
   ```

   **Rebuild index after changing extraction or chunking** (e.g. extraction normalization, pre-chunk filter):
   ```bash
   rm -rf data/processed indexes
   make ingest
   make build_index
   ```
   Then run `python scripts/test_extraction_normalization.py` to confirm chunk text has no broken prefixes (e.g. "tailored" not "ilored").

## Directory layout (after running)

- **data/**  
  - `processed/sources.jsonl` — extracted text per source  
  - `processed/chunks.jsonl` — chunk records (chunk_id, source_id, text, start_char, end_char)  
  - `manifest.csv` — metadata per source (source_id, title, authors, year, type, venue, link_or_doi, relevance_note, local_path)  
  - `eval_queries.json` — 20 evaluation queries (10 direct, 5 synthesis, 5 edge/ambiguity)

- **indexes/**  
  - `faiss.index` — vector index  
  - `docstore.json` — chunk_id / source_id / text mapping  
  - `bm25/` — BM25 index (meta, corpus, pickle)

- **logs/runs/**  
  - `phase2_runs.jsonl` — one JSON object per query run (timestamp, query_id, query_text, retrieval_method, top_k, retrieved_chunks, generated_answer, citations, model_name, prompt_version)

- **metrics/**  
  - `phase2_eval_summary.json` — aggregate metrics (groundedness/faithfulness heuristic, citation_precision)  
  - `phase2_eval_per_query.csv` — per-query breakdown

## Commands

### One-command demo (build index if needed, run one query, print answer + Evidence + References + APA→chunk mapping)

```bash
make phase2_demo
```

Or:

```bash
python -m prp run_demo
```

This will:
1. Ingest PDFs from `data/raw/` and create `data/processed/sources.jsonl` and `data/manifest.csv` if not present.
2. Chunk into `data/processed/chunks.jsonl` if not present.
3. Build FAISS + BM25 indexes in `indexes/` if not present.
4. Run one demo query, print retrieved chunk IDs, answer, Evidence, References, APA→chunk mapping, and append a log entry to `logs/runs/phase2_runs.jsonl`.

### Build index only

```bash
make build_index
# or
python scripts/build_index.py
```

### Run a single query

```bash
python -m prp query "Your question here"
# or
python scripts/run_query.py "Your question here" [optional_query_id]
```

**Output format:** Answer (synthesis only, no citations) + Evidence (quotes with APA-style `(Author et al., Year)` citations) + References and Reference-to-Chunk Mapping. Use `SHOW_REFERENCES=0` to hide References.

**USE_LLM:** `USE_LLM=1` (default) uses the LLM for Answer synthesis; `USE_LLM=0` uses extractive summarization. No hybrid fallback.

### Run stress tests

```bash
python scripts/run_stress_tests.py
```

Runs five stress tests (PNAS pipeline, design choices, long-term adherence + limitations, hard formatting constraint, conflict evidence), writes results to `logs/stress_tests_v3.txt`, and prints a summary (Not found count, validation failures, evidence mix distribution, top recurring sources).

### Run evaluation set (20 queries)

```bash
make run_eval
# or
python scripts/run_eval.py
```

Writes:
- `metrics/phase2_eval_summary.json`
- `metrics/phase2_eval_per_query.csv`

## Evaluation metrics

- **Groundedness/faithfulness (heuristic):** fraction of answer words that appear in retrieved text.
- **Citation precision:** fraction of citations that resolve to actual retrieved chunk text in `chunks.jsonl`.

## Enhancement implemented

**Hybrid retrieval:** BM25 (rank_bm25) + vector (FAISS). Results are merged using **reciprocal rank fusion (RRF)**. No API keys required; embeddings use `sentence-transformers/all-MiniLM-L6-v2` locally.

**Two-pass retrieval:** For general queries (non-compare), retrieval runs in two passes. Pass 1 retrieves answer-focused chunks. Pass 2 runs a modified query (original + limitations/uncertainty/bias/generalizability/adherence keywords) to find limitation-related evidence. Results are merged with deduplication and per-source cap. The "Limitations / uncertainty" section is written **only** from Pass 2 chunks; if none are found, the answer states "No limitation-related evidence retrieved." Logs show which chunks came from Pass 1 vs Pass 2.

**Evidence mix source of truth:** Evidence type (RCT, observational, systematic review, scoping review, other) is determined by: (1) manifest `type` when present (e.g. `RCT`, `review`), (2) chunk text classification via `evidence_type_label()` (keywords: "randomized controlled trial", "RCT", "systematic review", etc.). A warning is logged when retrieved sources have missing `study_type` and are counted as "other." Run `python tests/test_evidence_mix_bermingham.py` to assert Bermingham (Nature Medicine RCT) counts as RCT.

**Source diversification:** When selecting the final `top_k` chunks, retrieval caps the number of chunks per `source_id` (`MAX_PER_SOURCE` in `prp/config.py`, default 2). This avoids collapsing to a single document and improves coverage and groundedness. Remaining slots are filled from the next best sources by RRF score. To assert diversification, run `python scripts/test_retrieval_diversification.py` (requires indexes). Retrieved chunk IDs and `source_id`s are logged at INFO for debugging.

**Query-aware relevance:** After RRF + diversification, chunks are re-ranked by a weighted combination of RRF score and query–chunk embedding similarity (`RELEVANCE_RRF_WEIGHT` + `RELEVANCE_SIM_WEIGHT` in `prp/config.py`). Chunks with cosine similarity below `RELEVANCE_MIN_SIMILARITY` (default 0.25) are deprioritized when enough above-threshold chunks exist. Chunks containing unrelated disease terms (e.g. "cancer", "chemotherapy") are downweighted when the query does not mention cancer (`UNRELATED_CONTEXT_PENALTY`). Threshold 0.25 was chosen so that on sample queries ("effectiveness and limitations", general personalized-nutrition questions) irrelevant surface-keyword matches (e.g. cancer taste/smell) drop out of top 5. Run `python scripts/test_relevance_effectiveness_limitations.py` to assert Nature Medicine RCT appears in top 5 and cancer chunks do not when query doesn't mention cancer.

**Filters: funding/ack + bibliography.** Before chunking, lines likely to be acknowledgements, funding, conflicts of interest, or publisher/metadata boilerplate are removed (e.g. “supported by”, “funded by”, “grant”, “acknowledg”, “Received:”, “frontiersin.org”). This keeps chunks grounded and avoids citing non-content text. At retrieval time, chunks from References/Bibliography sections are also filtered (headings, citation patterns, semicolon density, author-year patterns, numbered refs). The filters are conservative so legitimate methods/results are not removed. After rebuilding, run `python scripts/check_chunks_ack_filter.py` to confirm ack/funding counts are 0 or near 0. Run `pytest tests/test_filters_and_evidence.py` to verify filters.

## Trust behavior

- **No invented citations:** Every citation is a retrieved chunk; none are fabricated.
- **Unsupported strong claims:** For queries that assert strong claims (e.g. “cures”, “guarantee”, “proven”), the system refuses to support the claim unless the retrieved text explicitly contains the claim language. Otherwise it returns a refusal message and lists related evidence as “not supporting the claim.”
- If retrieval returns no or low-confidence chunks, the answer states that no evidence was found.
- If conflicting evidence is detected (heuristic), the answer explicitly flags it and cites both sides.
- Citations are in the form `(source_id, chunk_id)` and resolve to entries in `data/processed/chunks.jsonl`.

## Phase 1

Phase 1 files are left unchanged; this README and the Phase 2 code live alongside them.
