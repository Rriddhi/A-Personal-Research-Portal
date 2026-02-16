#!/usr/bin/env python3
"""
Regression test: retrieval source diversification.
Asserts that for a compare-style query, retrieved_chunks include ≥2 distinct source_ids.
Requires: make ingest && make build_index (indexes must exist).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QUERY = (
    "Compare biomarker-based personalized nutrition vs survey-based personalized nutrition."
)
MIN_DISTINCT_SOURCES = 2


def main() -> int:
    from prp.run import run_query

    try:
        result = run_query(QUERY, query_id="test_diversification")
    except FileNotFoundError as e:
        print("SKIP: Index not found. Run: make ingest && make build_index")
        print(str(e))
        return 0
    except ModuleNotFoundError as e:
        print("SKIP: Missing dependency.", str(e))
        return 0

    chunks = result.get("retrieved_chunks") or []
    source_ids = [c.get("source_id") or "" for c in chunks]
    distinct = len(set(s for s in source_ids if s))
    assert distinct >= MIN_DISTINCT_SOURCES, (
        f"Expected ≥{MIN_DISTINCT_SOURCES} distinct source_ids in retrieved_chunks, got {distinct}. "
        f"source_ids=%s" % source_ids
    )
    print("PASS: Retrieved chunks have ≥%d distinct source_ids: %s" % (MIN_DISTINCT_SOURCES, list(set(source_ids))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
