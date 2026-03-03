#!/usr/bin/env python3
"""Sanity checks for Phase 3: manifest validation, evidence table, eval_stats (no index required)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    from prp.manifest_validate import validate_manifest, autofill_missing_fields, _load_schema
    from prp.artifacts import build_evidence_table, render_evidence_table_md, render_evidence_table_csv
    from prp.eval_stats import bootstrap_ci, paired_bootstrap_diff

    # Manifest
    schema = _load_schema()
    assert "required" in schema and "optional" in schema
    filled = autofill_missing_fields({"source_id": "s1", "title": "T"}, schema)
    assert "unknown_reason" in filled or len(schema["required"]) == 0
    print("manifest_validate: OK")

    # Evidence table
    rows = build_evidence_table(
        "Claim one. Claim two.",
        [("s1", "s1_c01")],
        [{"chunk_id": "s1_c01", "source_id": "s1", "text": "Evidence.", "rrf_score": 0.5}],
    )
    assert isinstance(rows, list)
    assert "|" in render_evidence_table_md(rows)
    assert "Claim" in render_evidence_table_csv(rows)
    print("artifacts: OK")

    # Bootstrap
    m, lo, hi = bootstrap_ci([1.0, 2.0, 3.0], n=100)
    assert lo <= m <= hi
    d, dlo, dhi = paired_bootstrap_diff([1.0, 2.0], [2.0, 3.0], n=100)
    assert dlo <= d <= dhi
    print("eval_stats: OK")

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
