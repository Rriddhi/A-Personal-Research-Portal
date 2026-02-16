#!/usr/bin/env python3
"""Run evaluation: eval queries, write per_query_metrics.json, aggregate_metrics.json, evaluation_summary.md."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.evaluate import run_evaluation, write_phase2_metrics, write_evaluation_summary_md, load_eval_queries

if __name__ == "__main__":
    agg, per_query = run_evaluation()
    write_phase2_metrics(per_query, agg)
    failure_cases = [{"query_id": r["query_id"], "query_text": "", "issue": "Low scores"} for r in per_query[:3]]
    qmap = {q["query_id"]: q for q in load_eval_queries()}
    for i, fc in enumerate(failure_cases):
        fc["query_text"] = qmap.get(fc["query_id"], {}).get("query_text", "")
    write_evaluation_summary_md(agg, None, failure_cases)
    print("Evaluation complete. Mean scores:")
    for k, v in sorted(agg.items()):
        print(f"  {k}: {v}")
    print("Output: metrics/per_query_metrics.json, metrics/aggregate_metrics.json, metrics/evaluation_summary.md")
