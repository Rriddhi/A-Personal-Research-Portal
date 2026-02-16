#!/usr/bin/env python3
"""
Unit test for question-conditioned extraction: run the privacy/governance query and assert
the answer is on-topic (privacy + governance terms) and does not contain irrelevant phrases.
Requires: make ingest && make build_index (indexes must exist).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Privacy-related phrases the answer should contain at least one of
PRIVACY_PHRASES = ("privacy", "security", "data")

# Governance/framework phrases the answer should contain at least one of
GOVERNANCE_PHRASES = ("framework", "lifecycle", "monitoring", "stakeholders")

# Irrelevant phrase that must NOT appear unless the query mentions it
IRRELEVANT_PHRASE = "calculation errors"

QUERY = (
    "In personalized nutrition systems, what are the major data privacy risks "
    "and what governance frameworks propose mitigations?"
)


def main() -> int:
    from prp.run import run_query

    try:
        result = run_query(QUERY, query_id="test_privacy_governance")
    except FileNotFoundError as e:
        print("SKIP: Index not found. Run: make ingest && make build_index")
        print(str(e))
        return 0  # skip in CI when indexes not built
    except ModuleNotFoundError as e:
        print("SKIP: Missing dependency (e.g. sentence_transformers). Install requirements.")
        print(str(e))
        return 0

    answer = (result.get("answer") or "").lower()
    query_lower = QUERY.lower()

    # Assert: answer contains at least one privacy-related phrase
    has_privacy = any(p in answer for p in PRIVACY_PHRASES)
    assert has_privacy, (
        f"Answer must contain at least one of {PRIVACY_PHRASES}. Got (excerpt): {answer[:500]}..."
    )

    # Assert: answer contains at least one governance term
    has_governance = any(g in answer for g in GOVERNANCE_PHRASES)
    assert has_governance, (
        f"Answer must contain at least one of {GOVERNANCE_PHRASES}. Got (excerpt): {answer[:500]}..."
    )

    # Assert: answer does NOT contain irrelevant phrase unless query mentions it
    if IRRELEVANT_PHRASE not in query_lower and IRRELEVANT_PHRASE in answer:
        raise AssertionError(
            f"Answer must not contain irrelevant phrase '{IRRELEVANT_PHRASE}'. Got (excerpt): {answer[:500]}..."
        )

    print("PASS: Answer is on-topic (privacy + governance) and free of irrelevant phrase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
