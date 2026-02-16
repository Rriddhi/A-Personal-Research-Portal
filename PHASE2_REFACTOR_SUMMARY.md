# Phase 2 Refactor Summary

## Files Modified

### Core Generation & Output
- **prp/generate.py**
  - Replaced output sections (Direct Answer, What the Corpus DOES Say, etc.) with Phase 2 structure: Answer + Evidence + References
  - Added `parse_citations_from_answer()`, `validate_citations_against_retrieved()` for trust validation
  - Added `_format_phase2_output()` for standardized output
  - Refactored `_extractive_synthesis()` to produce 250–500 word Answer with inline (source_id, chunk_id) citations, min 3 Evidence snippets, APA References
  - Updated `_format_not_found_detailed_answer()`, `_generate_claim_validation_response()`, `_generate_list_5_claims_response()` to use Phase 2 format
  - Updated `_generate_with_llm()` with Phase 2 prompt and citation validation
  - Added citation validation loop: if mismatch → regenerate

### Run Pipeline
- **prp/run.py**
  - Added `mode` parameter: `"llm_only"` | `"baseline"` | `"enhanced"`
  - LLM-only: no retrieval, LLM generates
  - Baseline: hybrid BM25+FAISS RRF, two-pass
  - Enhanced: expanded retrieval + rerank
  - Per-query logs now saved to `logs/runs/<timestamp>.json`

### Evaluation
- **prp/evaluate.py**
  - Implemented 8 metrics: context_precision, context_recall, faithfulness, citation_precision, answer_relevance, coherence, conciseness, artifact_readiness
  - LLM-based scoring (1–4) for faithfulness, answer_relevance, coherence, conciseness, artifact_readiness
  - Keyword-based fallback for context precision/recall when no LLM
  - Added `run_evaluation_single_mode()`, `write_phase2_metrics()`, `write_evaluation_summary_md()`

### Config & Utilities
- **prp/config.py**
  - Added `ANSWER_MIN_WORDS`, `ANSWER_MAX_WORDS`, `MIN_EVIDENCE_SNIPPETS`
  - Added `PER_QUERY_METRICS_JSON`, `AGGREGATE_METRICS_JSON`, `EVALUATION_SUMMARY_MD`

- **prp/utils.py**
  - Added `save_run_to_file(record, metrics)` for `logs/runs/<timestamp>.json`

- **scripts/run_query.py**
  - Updated to handle Phase 2 output format

- **scripts/run_eval.py**
  - Updated to write per_query_metrics.json, aggregate_metrics.json, evaluation_summary.md

- **prp/__main__.py**
  - Added `query`, `eval`, `eval_full` subcommands

## Files Added

- **scripts/run_evaluation_full.py** – Runs Baseline + Enhanced (+ optional LLM-only) modes, compares aggregate metrics, writes evaluation_summary.md with 3 failure cases

## How to Run

### Single Query (retrieval + answer + logging)
```bash
python -m prp query "What evidence exists for personalized health recommendations?"
# Or via script:
python scripts/run_query.py "What evidence exists for personalized health recommendations?"
```

### Evaluation (20-query set)
```bash
# Basic evaluation
python scripts/run_eval.py
# Or:
python -m prp eval

# Full evaluation (Baseline + Enhanced comparison)
python scripts/run_evaluation_full.py
# Or:
python -m prp eval_full
```

### Demo (build index if needed, run one query)
```bash
python -m prp run_demo
```

## Output Structure (Phase 2 Compliant)

```
----------------------------------------
Answer:
----------------------------------------
(250–500 words, synthesized, inline (source_id) or (source_id, chunk_id), academic tone)

----------------------------------------
Evidence:
----------------------------------------
1. "Exact quote..." (source_id, chunk_id)
   → This supports the claim that...

2. "Another sentence..." (source_id, chunk_id)
   → This indicates...

(Minimum 3 snippets)

----------------------------------------
References:
----------------------------------------
(APA format from manifest metadata, only cited sources)
```

## Trust Behavior

- Parse Answer for all (source_id, chunk_id)
- Verify each citation exists in retrieved chunks
- If mismatch → regenerate answer
- Never fabricate citations
- If evidence absent → explicitly state "The corpus does not contain direct evidence that..."

## Metric Outputs

- **per_query_metrics.json** – Per-query scores for all 8 metrics
- **aggregate_metrics.json** – Mean scores across queries
- **evaluation_summary.md** – Mean scores, 3 failure cases, Baseline vs Enhanced comparison

## Acceptance Criteria Met

- ✓ One command runs retrieval + answer + logging
- ✓ Citations resolve to real chunk IDs
- ✓ 20-query evaluation set supported
- ✓ At least one enhancement (Enhanced mode) shows measurable improvement when run
- ✓ Report includes 3 failure cases
- ✓ No ingestion pipeline changes
- ✓ No UI changes
