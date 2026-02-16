#!/usr/bin/env python3
"""Run a single query: retrieve + generate, print answer and citations, append log."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.run import run_query


def render_response(result: dict, debug: bool = False) -> None:
    """
    Single controlled output: exactly ONE Answer block.
    Normal: Query ID + Answer (with APA references). No chunk_ids.
    Debug: + retrieval_method, top_k, condition_found_lexically, chunk_ids, citations.
    """
    answer = result.get("answer", "")
    stripped = answer.strip()
    # Phase 2 format already has Answer/Evidence/References structure; don't duplicate
    if "----------------------------------------" in stripped and "Answer:" in stripped:
        answer = stripped
    else:
        if stripped.lower().startswith("answer:"):
            stripped = stripped[7:].lstrip()
        answer = "Answer:\n" + stripped

    print("Query ID:", result.get("query_id", ""))
    if debug:
        print("[DEBUG] retrieval_method:", result.get("retrieval_method", "unknown"))
        print("[DEBUG] top_k:", result.get("top_k", "?"))
        print("[DEBUG] condition_found_lexically:", result.get("condition_found_lexically", False))
        print("[DEBUG] chunk_ids:", [c["chunk_id"] for c in result.get("retrieved_chunks", [])])
        print("[DEBUG] citations:", result.get("citations", []))

    print("\n" + answer)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--debug"]
    debug = "--debug" in sys.argv or os.environ.get("PRP_DEBUG", "").strip().lower() in ("1", "true", "yes")
    query = args[0] if args else "What evidence exists for personalized health recommendations?"
    query_id = args[1] if len(args) > 1 else None
    result = run_query(query, query_id=query_id)
    render_response(result, debug=debug)
