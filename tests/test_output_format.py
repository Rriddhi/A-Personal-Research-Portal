"""Tests for Phase 2 output format: Answer (no citations), Evidence (quotes + chunk IDs), References."""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _extract_answer_section(output: str) -> str:
    """Extract Answer section from Phase 2 output."""
    idx_a = output.find("Answer:")
    idx_e = output.find("Evidence:")
    if idx_a < 0 or idx_e <= idx_a:
        return output
    return output[idx_a + 7 : idx_e].strip()


def _extract_evidence_section(output: str) -> str:
    """Extract Evidence section from Phase 2 output."""
    idx_e = output.find("Evidence:")
    idx_r = output.find("References:")
    if idx_e < 0:
        return ""
    return output[idx_e : idx_r] if idx_r > idx_e else output[idx_e:]


def _extract_references_section(output: str) -> str:
    """Extract References section from Phase 2 output."""
    idx_r = output.find("References:")
    if idx_r < 0:
        return ""
    return output[idx_r:]


INTERNAL_CITATION_RE = re.compile(r"\(([A-Za-z0-9_\-\.]+_c\d+)\)")
APA_CITATION_RE = re.compile(r"\([A-Za-z][^)]*?,?\s*(?:\d{4}|n\.d\.)[^)]*\)", re.IGNORECASE)


def test_answer_contains_no_citations():
    """Answer must not contain (chunk_id) or (Author et al., Year) patterns."""
    from prp.generate import _format_phase2_output
    from prp.citations import references_apa_sorted

    answer_text = "Evidence suggests personalized nutrition improves outcomes. Limitations exist."
    evidence_items = [
        ('"Quote one."', "sid1", "sid1_c0001", "Supports: outcomes claim"),
        ('"Quote two."', "sid2", "sid2_c0002", "Supports: limitations"),
    ]
    ref_ids = ["sid1", "sid2"]
    output = _format_phase2_output(answer_text, evidence_items, ref_ids)

    answer_section = _extract_answer_section(output)
    internal_matches = list(INTERNAL_CITATION_RE.finditer(answer_section))
    apa_matches = list(APA_CITATION_RE.finditer(answer_section))
    assert len(internal_matches) == 0, f"Answer should have no internal citations, found: {internal_matches}"
    assert len(apa_matches) == 0, f"Answer should have no APA citations, found: {apa_matches}"


def test_evidence_citations_match_format():
    """Evidence citations must be (source_id_chunk_id) format, no .pdf or hashes."""
    from prp.generate import normalize_chunk_citation_for_display

    assert normalize_chunk_citation_for_display(
        "s41746-025-01900-y.pdf_1e64f98603e3_c0049", "s41746-025-01900-y"
    ) == "(s41746-025-01900-y_c0049)"
    assert "_c" in normalize_chunk_citation_for_display("x_c0001", "x")
    assert ".pdf" not in normalize_chunk_citation_for_display(
        "doc.pdf_abc123_c0001", "doc"
    )


def test_evidence_has_supports_format():
    """Each Evidence item must include 'Supports:' in explanation."""
    from prp.generate import _format_phase2_output

    answer_text = "Synthesis."
    evidence_items = [
        ('"Quote."', "sid1", "sid1_c0001", "Supports: main claim"),
    ]
    output = _format_phase2_output(answer_text, evidence_items, ["sid1"])
    evidence_section = _extract_evidence_section(output)
    assert "Supports:" in evidence_section


def test_references_include_only_cited_sources():
    """Every source_id in Evidence must appear in References."""
    from prp.generate import _format_phase2_output
    from prp.citations import format_in_text_apa

    evidence_items = [
        ('"Q1."', "source_a", "source_a_c0001", "Supports: claim 1"),
        ('"Q2."', "source_a", "source_a_c0002", "Supports: claim 2"),
        ('"Q3."', "source_b", "source_b_c0001", "Supports: claim 3"),
    ]
    ref_ids = ["source_a", "source_b"]
    output = _format_phase2_output(
        "Answer text.", evidence_items, ref_ids, show_references=True
    )
    refs_section = _extract_references_section(output)
    for sid in ref_ids:
        assert sid in refs_section or format_in_text_apa(sid).split(",")[0] in refs_section


def test_strip_all_citations_removes_internal_and_apa():
    """strip_all_citations_from_text removes both (chunk_id) and (Author, Year)."""
    from prp.generate import strip_all_citations_from_text

    text = "Claim one (jmir_c0001). Claim two (Smith et al., 2024)."
    cleaned = strip_all_citations_from_text(text)
    assert "jmir_c0001" not in cleaned
    assert "Smith et al." not in cleaned
