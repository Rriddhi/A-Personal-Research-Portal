#!/usr/bin/env python3
"""
Regression test: queries like "effectiveness and limitations" must NOT cite
methodology boilerplate (e.g. "Stage 5: Collating…") as evidence.
Asserts that no chunk in retrieved_chunks contains Stage N + Collating/PRISMA-style boilerplate.
Requires: make ingest && make build_index.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QUERY = "effectiveness and limitations"
# Chunks that are predominantly methodology boilerplate must not appear in retrieval
BOILERPLATE_PHRASES = (
    "Stage 5",
    "Stage 4",
    "Stage 3",
    "Stage 2",
    "Stage 1",
    "Collating",
    "collating, summarizing",
    "PRISMA",
)


def main() -> int:
    from prp.run import run_query

    try:
        result = run_query(QUERY, query_id="test_no_boilerplate_citations")
    except FileNotFoundError as e:
        print("SKIP: Index not found. Run: make ingest && make build_index")
        print(str(e))
        return 0
    except ModuleNotFoundError as e:
        print("SKIP: Missing dependency.", str(e))
        return 0

    chunks = result.get("retrieved_chunks") or []
    for c in chunks:
        text = (c.get("text") or "").strip()
        for phrase in BOILERPLATE_PHRASES:
            if phrase in text:
                # Chunk should have been filtered by is_boilerplate; fail if we still see it
                print(
                    "FAIL: Retrieved chunk contains methodology boilerplate '%s'. "
                    "chunk_id=%s source_id=%s (excerpt): ...%s..."
                    % (phrase, c.get("chunk_id"), c.get("source_id"), text[:200])
                )
                return 1

    print("PASS: No methodology boilerplate (Stage 5: Collating, etc.) cited for query '%s'." % QUERY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
