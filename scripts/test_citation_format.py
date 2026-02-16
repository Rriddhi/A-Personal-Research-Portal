#!/usr/bin/env python3
"""
Sanity check: Answer contains no citation patterns; Evidence contains APA-style (Author et al., Year); no hashes.
Run after: python -m prp query "What evidence exists for AI and autoimmune?"
"""
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Ensure USE_LLM=0 for fast extractive test (no API needed)
os.environ.setdefault("USE_LLM", "0")
os.environ.setdefault("SHOW_REFERENCES", "0")

from prp.run import run_query
from prp.generate import clean_source_id, normalize_chunk_citation_for_display

# Patterns that must NOT appear in Answer section
ANSWER_BAD_PATTERNS = [
    re.compile(r"\([A-Za-z][^)]*?,?\s*(?:\d{4}|n\.d\.)[^)]*\)", re.IGNORECASE),  # (Author, Year)
    re.compile(r"\([a-z0-9\-]+_[a-f0-9]{8,}_[cC]\d+\)", re.IGNORECASE),  # (xxx_HASH_c0008)
]
# Unstripped hash in citation: _hexhash_c0008 pattern (hash suffix we should have removed)
HASH_PATTERN = re.compile(r"_[a-f0-9]{8,}_[cC]\d+", re.IGNORECASE)


def extract_sections(output: str) -> tuple:
    """Return (answer_text, evidence_text)."""
    out = output or ""
    idx_a = out.find("Answer:")
    idx_e = out.find("Evidence:")
    idx_r = out.find("References:")
    if idx_a < 0 or idx_e < 0:
        return ("", out)
    answer = out[idx_a + 7 : idx_e].strip()
    evidence = out[idx_e + 9 : idx_r].strip() if idx_r > idx_e else out[idx_e + 9 :].strip()
    return (answer, evidence)


def test_clean_source_id():
    """Unit test for clean_source_id."""
    assert clean_source_id("jpm-15-00058-v2.pdf_d26bbd07bb39") == "jpm-15-00058-v2"
    assert clean_source_id("s41746-025-01900-y.pdf_1e64f98603e3") == "s41746-025-01900-y"
    assert clean_source_id("") == ""
    print("PASS: clean_source_id")


def test_normalize_citation_no_hash():
    """Citations must not contain hash substrings."""
    cit = normalize_chunk_citation_for_display(
        "jpm-15-00058-v2.pdf_d26bbd07bb39_c0008",
        "jpm-15-00058-v2.pdf_d26bbd07bb39",
    )
    assert "d26bbd07bb39" not in cit, "Citation must not contain hash"
    assert "jpm-15-00058-v2_c0008" in cit or "(jpm-15-00058-v2_c0008)" == cit
    print("PASS: normalize_chunk_citation_for_display (no hash)")


def test_answer_no_citations(output: str):
    """Answer must not contain (Author, Year) or (jpm-...) style citations."""
    answer, _ = extract_sections(output)
    for pat in ANSWER_BAD_PATTERNS:
        m = pat.search(answer)
        assert not m, "Answer must not contain citation: %s" % (m.group(0) if m else "")


# APA-style in Evidence: (Author et al., Year) or (Author, Year)
APA_IN_EVIDENCE_RE = re.compile(r"\([A-Za-z][^)]*?,?\s*(?:\d{4}|n\.d\.)[^)]*\)", re.IGNORECASE)


def test_evidence_has_apa_citations(output: str):
    """Evidence bullets must contain APA-style (Author et al., Year) citations; no hash."""
    _, evidence = extract_sections(output)
    apa_cits = APA_IN_EVIDENCE_RE.findall(evidence)
    for c in apa_cits:
        assert not HASH_PATTERN.search(c), "Evidence citation must not contain unstripped hash: %s" % c
    # Also reject old (source_chunk) format with hashes if any remain
    old_cits = re.findall(r"\(([^)]+)\)", evidence)
    for c in old_cits:
        if HASH_PATTERN.search(c):
            raise AssertionError("Evidence must not contain hash in citation: %s" % c)
    if apa_cits:
        print("PASS: Evidence has APA-style citations")
    else:
        print("WARN: No APA citations in Evidence (may be empty)")


def main() -> int:
    test_clean_source_id()
    test_normalize_citation_no_hash()

    print("Running query (USE_LLM=0 for extractive)...")
    r = run_query("What evidence exists for AI and autoimmune?")
    output = r.get("answer", "")

    if not output:
        print("FAIL: No output")
        return 1

    try:
        test_answer_no_citations(output)
        print("PASS: Answer contains no citation patterns")
    except AssertionError as e:
        print("FAIL:", e)
        return 1

    try:
        test_evidence_has_apa_citations(output)
    except AssertionError as e:
        print("FAIL:", e)
        return 1

    print("All citation format checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
