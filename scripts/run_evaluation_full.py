#!/usr/bin/env python3
"""
Phase 2 full evaluation: run LLM-only, Baseline RAG, Enhanced RAG modes.
Output: per_query_metrics.json, aggregate_metrics.json, evaluation_summary.md with comparison.
"""
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.config import (
    METRICS_DIR,
    PER_QUERY_METRICS_JSON,
    AGGREGATE_METRICS_JSON,
    EVALUATION_SUMMARY_MD,
    EVAL_QUERIES_JSON,
    ensure_dirs,
)
from prp.run import run_query
from prp.evaluate import (
    run_evaluation_single_mode,
    write_phase2_metrics,
    write_evaluation_summary_md,
    load_eval_queries,
    compute_context_precision,
    compute_faithfulness,
    compute_citation_precision,
    compute_answer_relevance,
)
from prp.utils import logger


def _run_fn_for_mode(mode: str):
    def fn(query_text: str, query_id: str | None):
        return run_query(query_text, query_id=query_id, mode=mode)
    return fn


def _identify_failure_cases(per_query_baseline: list, queries: list) -> list:
    """Find 3 representative failure cases: lowest aggregate score."""
    qmap = {q["query_id"]: q for q in queries}
    scored = []
    for row in per_query_baseline:
        qid = row["query_id"]
        avg = (
            row.get("context_precision", 0) + row.get("faithfulness", 0)
            + row.get("citation_precision", 0) + row.get("answer_relevance", 0)
        ) / 4
        scored.append((avg, row, qmap.get(qid, {})))
    scored.sort(key=lambda x: x[0])
    failures = []
    for _, row, q in scored[:3]:
        failures.append({
            "query_id": row["query_id"],
            "query_text": q.get("query_text", ""),
            "issue": f"Low scores: context_precision={row.get('context_precision', 0):.2f}, "
                     f"faithfulness={row.get('faithfulness', 0):.2f}",
        })
    return failures


def main():
    ensure_dirs()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_eval_queries(EVAL_QUERIES_JSON)

    baseline_per, baseline_agg = run_evaluation_single_mode(
        "baseline",
        _run_fn_for_mode("baseline"),
        EVAL_QUERIES_JSON,
    )
    write_phase2_metrics(
        baseline_per,
        baseline_agg,
        PER_QUERY_METRICS_JSON,
        AGGREGATE_METRICS_JSON,
    )

    enhanced_per = None
    enhanced_agg = None
    if os.environ.get("RUN_ENHANCED", "1") == "1":
        try:
            enhanced_per, enhanced_agg = run_evaluation_single_mode(
                "enhanced",
                _run_fn_for_mode("enhanced"),
                EVAL_QUERIES_JSON,
            )
            # Write enhanced metrics to separate files
            write_phase2_metrics(
                enhanced_per,
                enhanced_agg,
                METRICS_DIR / "per_query_metrics_enhanced.json",
                METRICS_DIR / "aggregate_metrics_enhanced.json",
            )
        except Exception as e:
            logger.warning("Enhanced mode failed (need OPENAI_API_KEY?): %s", e)

    llm_only_agg = None
    if os.environ.get("RUN_LLM_ONLY", "0") == "1":
        try:
            _, llm_only_agg = run_evaluation_single_mode(
                "llm_only",
                _run_fn_for_mode("llm_only"),
                EVAL_QUERIES_JSON,
            )
        except Exception as e:
            logger.warning("LLM-only mode failed: %s", e)

    failure_cases = _identify_failure_cases(baseline_per, queries)
    write_evaluation_summary_md(
        baseline_agg,
        enhanced_agg or llm_only_agg,
        failure_cases,
        EVALUATION_SUMMARY_MD,
    )

    print("Phase 2 Evaluation Complete")
    print("=" * 50)
    print("Baseline RAG aggregate metrics:")
    for k, v in sorted(baseline_agg.items()):
        print(f"  {k}: {v}")
    if enhanced_agg:
        print("\nEnhanced RAG aggregate metrics:")
        for k, v in sorted(enhanced_agg.items()):
            print(f"  {k}: {v}")
    print(f"\nOutput: {PER_QUERY_METRICS_JSON}, {AGGREGATE_METRICS_JSON}, {EVALUATION_SUMMARY_MD}")


if __name__ == "__main__":
    main()
