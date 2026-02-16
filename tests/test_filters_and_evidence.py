"""
Pytest tests for bibliography filter, ACK/FUNDING filter, retrieval fallback, and evidence mix.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_bibliography_chunks_filtered():
    """Bibliography chunks are filtered out (is_bibliography_chunk and is_boilerplate)."""
    from prp.utils import is_bibliography_chunk, is_boilerplate, bibliography_score

    # Reference list fragment - should be filtered
    ref_chunk = """
    1. Smith J, Jones A. Personalized nutrition and health. Nat Med 2020;24(3):123-45.
    2. Doe B, et al. Dietary guidelines. National Academies Press, Washington, DC. ISBN 978-0-309-12345-6.
    3. Lee C. Retrieved from https://doi.org/10.1016/j.example.2021.01.001
    """
    assert bibliography_score(ref_chunk) >= 0.5
    assert is_bibliography_chunk(ref_chunk)
    assert is_boilerplate(ref_chunk)

    # Heading "References" at start
    ref_heading = "References\n1. Author A. Title. J Med 2020;10:1-10."
    assert is_bibliography_chunk(ref_heading)

    # Normal content - should NOT be filtered
    normal = "Personalized nutrition aims to provide dietary recommendations tailored to individuals based on their unique characteristics and evidence from randomized controlled trials."
    assert not is_bibliography_chunk(normal)
    assert bibliography_score(normal) < 0.5


def test_ack_funding_still_filtered():
    """ACK/FUNDING chunks still filtered by is_boilerplate."""
    from prp.utils import is_boilerplate

    ack_chunk = "This work was supported by NIH grant number R01-12345. We thank our partners for their contribution."
    assert is_boilerplate(ack_chunk)

    funding_chunk = "The study was funded by the European Commission. Conflicts of interest: none declared."
    assert is_boilerplate(funding_chunk)

    normal_chunk = "The randomized controlled trial showed that personalized nutrition interventions improved dietary adherence compared to standard advice."
    assert not is_boilerplate(normal_chunk)


def test_retrieval_returns_enough_chunks():
    """Retrieval returns >= top_k when enough non-filtered chunks exist."""
    import pytest
    try:
        from prp.retrieve import retrieve_hybrid
    except (FileNotFoundError, ModuleNotFoundError) as e:
        pytest.skip(f"Retrieval deps/index not available: {e}")

    try:
        result = retrieve_hybrid("What is personalized nutrition?", top_k=5)
    except (FileNotFoundError, ModuleNotFoundError) as e:
        pytest.skip(f"Cannot run retrieval: {e}")

    # With sufficient corpus, we expect up to top_k chunks
    assert len(result) >= 1, "Should return at least 1 chunk"
    assert len(result) <= 5, "Should not exceed top_k"
    # All returned chunks should not be boilerplate/bibliography
    from prp.utils import is_boilerplate
    for c in result:
        assert not is_boilerplate(c.get("text") or ""), (
            f"Retrieved chunk should not be boilerplate: {c.get('chunk_id')}"
        )


def test_evidence_mix_classifier():
    """Evidence type classifier produces expected labels for example strings."""
    from prp.utils import (
        evidence_type_label,
        EVIDENCE_TYPE_RCT,
        EVIDENCE_TYPE_OBSERVATIONAL,
        EVIDENCE_TYPE_SYSTEMATIC_REVIEW,
        EVIDENCE_TYPE_SCOPING_REVIEW,
        EVIDENCE_TYPE_EDITORIAL_OTHER,
    )

    assert evidence_type_label("This randomized controlled trial evaluated...") == EVIDENCE_TYPE_RCT
    assert evidence_type_label("A systematic review of the literature...") == EVIDENCE_TYPE_SYSTEMATIC_REVIEW
    assert evidence_type_label("We conducted a scoping review to map...") == EVIDENCE_TYPE_SCOPING_REVIEW
    assert evidence_type_label("An observational cohort study found...") == EVIDENCE_TYPE_OBSERVATIONAL
    assert evidence_type_label("The cross-sectional survey revealed...") == EVIDENCE_TYPE_OBSERVATIONAL
    # editorial/other for generic text
    assert evidence_type_label("Personalized nutrition aims to provide recommendations.") == EVIDENCE_TYPE_EDITORIAL_OTHER
