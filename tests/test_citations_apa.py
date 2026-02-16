"""Unit tests for APA citation formatting."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_format_in_text_apa():
    """APA in-text: (Author, Year) or (Author et al., Year)."""
    from prp.citations import format_in_text_apa, SOURCE_MAP

    if SOURCE_MAP:
        sid = list(SOURCE_MAP.keys())[0]
        apa = format_in_text_apa(sid)
        assert "(" in apa and ")" in apa
        assert "et al." in apa or "," in apa


def test_format_reference_apa():
    """APA reference: Author (Year). Title. Journal. URL. No source_id."""
    from prp.citations import format_reference_apa, SOURCE_MAP

    if SOURCE_MAP:
        sid = list(SOURCE_MAP.keys())[0]
        ref = format_reference_apa(sid)
        assert "(" in ref and ")." in ref
        assert "References:" not in ref
        assert sid not in ref
