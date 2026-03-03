"""
Compile evaluation results into outputs/eval/eval_summary.md: one comparison table
(bm25 vs faiss vs rrf vs rrf_guardrails) with mean ± 95% CI per metric.
"""
import json
from pathlib import Path
from typing import List, Optional

from .config import OUTPUTS_EVAL_DIR, EVAL_SUMMARY_MD_OUTPUT, ensure_dirs
from .eval_stats import bootstrap_ci


def _load_condition_metrics() -> dict:
    """Load per-condition metrics from outputs/eval/{condition}_metrics.json and _runs.jsonl."""
    out = {}
    for path in OUTPUTS_EVAL_DIR.glob("*_metrics.json"):
        cond = path.stem.replace("_metrics", "")
        with open(path, "r", encoding="utf-8") as f:
            out[cond] = {"aggregate": json.load(f), "per_query": []}
    for path in OUTPUTS_EVAL_DIR.glob("*_runs.jsonl"):
        cond = path.stem.replace("_runs", "")
        if cond not in out:
            out[cond] = {"aggregate": {}, "per_query": []}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out[cond]["per_query"].append(json.loads(line))
    return out


def compile_eval_summary_md(
    conditions_data: Optional[dict] = None,
    output_path: Optional[Path] = None,
    eval_queries_path: Optional[Path] = None,
) -> str:
    """
    Compile eval_summary.md: one table comparing the four retrieval methods.
    If conditions_data is None, load from OUTPUTS_EVAL_DIR.
    Returns path to written file.
    """
    ensure_dirs()
    output_path = output_path or EVAL_SUMMARY_MD_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = conditions_data or _load_condition_metrics()

    key_to_field = {
        "context_precision": "context_precision",
        "citation_precision": "citation_precision",
        "retrieval_diversity": "retrieval_diversity",
        "citation_coverage": "citation_coverage",
        "unsupported_claim_rate": "unsupported_claim_rate",
        "faithfulness_proxy_mean": "faithfulness_proxy_mean",
    }
    metric_keys = list(key_to_field.keys())

    lines = [
        "# Ablation comparison",
        "",
        "One row per retrieval method; each cell is mean ± 95% CI over eval queries. "
        "Re-run **Run ablations** to refresh.",
        "",
        "| Method | " + " | ".join(metric_keys) + " |",
        "|" + "|".join(["---"] * (len(metric_keys) + 1)) + "|",
    ]
    for cond in sorted(data.keys()):
        payload = data[cond]
        per_query = payload.get("per_query", [])
        cells = [cond]
        for field in metric_keys:
            vals = [r.get(field) for r in per_query]
            vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                if "diversity" in field:
                    cells.append(f"{mean:.2f} [{lo:.2f}, {hi:.2f}]")
                else:
                    cells.append(f"{mean:.4f} [{lo:.4f}, {hi:.4f}]")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def main():
    compile_eval_summary_md()
    print("Wrote", EVAL_SUMMARY_MD_OUTPUT)


if __name__ == "__main__":
    main()
