#!/usr/bin/env python3
"""Test that Bermingham (Nature Medicine RCT, doi 10.1038/s41591-024-02951-6) counts as RCT in Evidence mix."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.utils import evidence_type_label, EVIDENCE_TYPE_RCT


def test_bermingham_chunk_counts_as_rct():
    """Bermingham c0000 chunk text contains 'randomized controlled trial'; must classify as RCT."""
    # Actual chunk text from Bermingham (s41591-024-02951-6) - first chunk
    chunk_text = (
        "Effects of a personalized nutrition program on cardiometabolic health: "
        "a randomized controlled trial Large variability exists in people's responses to foods. "
        "However, the efficacy of personalized nutrition interventions remains unclear."
    )
    result = evidence_type_label(chunk_text)
    assert result == EVIDENCE_TYPE_RCT, (
        f"Bermingham chunk should classify as RCT, got: {result}"
    )


def test_bermingham_source_id_in_docstore():
    """If Bermingham is in docstore, evidence_type_label on its text must return RCT."""
    docstore_path = REPO / "indexes" / "docstore.json"
    if not docstore_path.exists():
        return  # skip if no index
    import json
    with open(docstore_path, "r", encoding="utf-8") as f:
        docstore = json.load(f)
    bermingham = [d for d in docstore if "s41591-024-02951-6" in (d.get("source_id") or "")]
    for d in bermingham:
        text = d.get("text") or ""
        if "randomized" in text.lower() or "randomised" in text.lower():
            tag = evidence_type_label(text)
            assert tag == EVIDENCE_TYPE_RCT, (
                f"Bermingham chunk {d.get('chunk_id')} should be RCT, got: {tag}"
            )
            return
    # If no Bermingham in docstore, test passes (corpus may differ)
    if not bermingham:
        return
    # Bermingham present but no RCT phrase in any chunk - still run label on first chunk
    text = bermingham[0].get("text") or ""
    tag = evidence_type_label(text)
    # At least one Bermingham chunk should be RCT (title/subtitle typically in c0000)
    assert tag == EVIDENCE_TYPE_RCT or any(
        evidence_type_label(c.get("text") or "") == EVIDENCE_TYPE_RCT for c in bermingham
    ), f"No Bermingham chunk classified as RCT; first chunk got: {tag}"


if __name__ == "__main__":
    test_bermingham_chunk_counts_as_rct()
    test_bermingham_source_id_in_docstore()
    print("OK: Bermingham evidence mix tests passed.")
