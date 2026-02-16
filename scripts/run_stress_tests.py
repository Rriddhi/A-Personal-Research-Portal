#!/usr/bin/env python3
"""
Stress test runner: runs a list of test queries, writes logs to logs/stress_tests_v3.txt.
Summary at end: Not found count, validation failures, evidence mix distribution, top sources.
"""
import sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.run import run_query
from prp.generate import _validate_format_constraint, _is_format_constraint_query

LOG_PATH = REPO / "logs" / "stress_tests_v3.txt"

STRESS_TESTS = [
    {
        "id": "stress_01_pnas",
        "query": "What is the PNAS pipeline from data to model to delivery to feedback?",
        "description": "PNAS pipeline end-to-end",
    },
    {
        "id": "stress_02_design",
        "query": "What design choices with headings are discussed for personalized nutrition systems?",
        "description": "Design choices with headings",
    },
    {
        "id": "stress_03_adherence",
        "query": "Long-term adherence and limitations of personalized health interventions",
        "description": "Long-term adherence + limitations",
    },
    {
        "id": "stress_04_format",
        "query": "List 5 claims about personalized nutrition. MUST have exactly one citation and an evidence snippet per claim.",
        "description": "Hard formatting constraint: 5 claims, exactly one citation + one evidence snippet each",
    },
    {
        "id": "stress_05_conflict",
        "query": "Evidence for and against effectiveness of personalized dietary recommendations",
        "description": "Conflict: evidence for and against effectiveness",
    },
]


def extract_evidence_mix(answer: str) -> dict:
    """Parse 'Evidence mix: RCT=1, observational=0, ...' into dict."""
    out = {}
    if "Evidence mix:" not in answer:
        return out
    part = answer.split("Evidence mix:")[-1].strip()
    for tok in part.split(","):
        tok = tok.strip()
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return out


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    not_found = 0
    validation_failures = 0
    evidence_mix_totals: Counter = Counter()
    source_counts: Counter = Counter()

    lines = ["=" * 60, "Stress tests v3", "=" * 60]

    for t in STRESS_TESTS:
        qid = t["id"]
        query = t["query"]
        desc = t["description"]
        lines.append(f"\n--- {qid}: {desc} ---")
        lines.append(f"Query: {query}")
        try:
            result = run_query(query, query_id=qid)
            answer = result["answer"]
            citations = result["citations"]

            if "Not found" in answer and len(citations) == 0:
                not_found += 1
                lines.append("[NOT FOUND]")

            if _is_format_constraint_query(query):
                valid, violations = _validate_format_constraint(answer, expected_per_claim=1)
                if not valid:
                    validation_failures += 1
                    lines.append(f"[VALIDATION FAILED] {violations}")

            mix = extract_evidence_mix(answer)
            for k, v in mix.items():
                evidence_mix_totals[k] += v

            for sid, _ in citations:
                if sid:
                    source_counts[sid] += 1

            lines.append("\n" + answer[:3000] + ("..." if len(answer) > 3000 else ""))
            lines.append("\n--- end ---")
        except Exception as e:
            lines.append(f"[ERROR] {e}")
            not_found += 1

    lines.append("\n" + "=" * 60)
    lines.append("SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Not found: {not_found}")
    lines.append(f"Validation failures (format constraint): {validation_failures}")
    lines.append(f"Evidence mix totals: {dict(evidence_mix_totals)}")
    top_sources = source_counts.most_common(10)
    lines.append(f"Top recurring sources: {top_sources}")

    out = "\n".join(lines)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\nLog written to {LOG_PATH}")


if __name__ == "__main__":
    main()
