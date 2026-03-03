"""
Phase 3 tests: session, export, quality gates.
"""
import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_validate_answer_package_valid():
    """Valid package with required keys passes."""
    from prp.quality import validate_answer_package
    pkg = {
        "answer": "Some answer.",
        "citations": [["sid1", "sid1_c0001"]],
        "citation_mapping": [{"apa": "(Author et al., 2020)", "source_id": "sid1", "chunk_id": "sid1_c0001"}],
        "retrieved_chunks": [{"chunk_id": "sid1_c0001", "source_id": "sid1", "text_preview": "preview"}],
        "metadata": {"query_id": "q1", "timestamp": "2026-01-01T00:00:00", "retrieval_method": "hybrid", "query_text": "What?"},
    }
    ok, errs = validate_answer_package(pkg)
    assert ok, errs
    assert len(errs) == 0


def test_validate_answer_package_missing_metadata():
    """Package with metadata missing required keys fails schema."""
    from prp.quality import validate_answer_package
    pkg = {
        "answer": "Some answer.",
        "citations": [],
        "citation_mapping": [],
        "retrieved_chunks": [],
        "metadata": {},
    }
    ok, errs = validate_answer_package(pkg)
    assert not ok
    assert any("query_id" in e or "metadata" in e for e in errs)


def test_validate_citations_in_retrieved_all_valid():
    """All citation chunk_ids in retrieved_chunks passes."""
    from prp.quality import validate_citations_in_retrieved
    pkg = {
        "citation_mapping": [{"chunk_id": "c1", "source_id": "s1", "apa": "(A, 2020)"}],
        "citations": [["s1", "c1"]],
        "retrieved_chunks": [{"chunk_id": "c1", "source_id": "s1", "text_preview": "x"}],
    }
    ok, errs = validate_citations_in_retrieved(pkg)
    assert ok, errs
    assert len(errs) == 0


def test_validate_citations_in_retrieved_fails_when_citation_not_retrieved():
    """Citation pointing to chunk_id not in retrieved_chunks fails."""
    from prp.quality import validate_citations_in_retrieved
    pkg = {
        "citation_mapping": [{"chunk_id": "ghost_chunk", "source_id": "s1", "apa": "(A, 2020)"}],
        "retrieved_chunks": [{"chunk_id": "c1", "source_id": "s1", "text_preview": "x"}],
    }
    ok, errs = validate_citations_in_retrieved(pkg)
    assert not ok
    assert any("ghost_chunk" in e for e in errs)


def test_validate_package_integration():
    """validate_package runs both schema and citation checks."""
    from prp.quality import validate_package
    pkg = {
        "answer": "Answer.",
        "citations": [["s1", "c1"]],
        "citation_mapping": [{"apa": "(A, 2020)", "source_id": "s1", "chunk_id": "c1"}],
        "retrieved_chunks": [{"chunk_id": "c1", "source_id": "s1", "text_preview": "p"}],
        "metadata": {"query_id": "q1", "timestamp": "t", "retrieval_method": "h", "query_text": "Q"},
    }
    ok, errs = validate_package(pkg)
    assert ok, errs


def test_session_create_append_reload():
    """Session create, append two runs (fixture data), reload and check count."""
    from prp.session import create_session, get_session
    from prp.config import SESSIONS_DIR
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid = create_session()
    path = SESSIONS_DIR / f"{sid}.jsonl"
    assert path.exists()
    # Append fixture lines (don't call run_query to avoid index/network)
    fixture = {
        "answer": "Fixture answer.",
        "citations": [["s1", "s1_c0001"]],
        "citation_mapping": [{"apa": "(A, 2020)", "source_id": "s1", "chunk_id": "s1_c0001"}],
        "retrieved_chunks": [{"chunk_id": "s1_c0001", "source_id": "s1", "text_preview": "preview"}],
        "query_id": "q1", "timestamp": "2026-01-01T00:00:00", "retrieval_method": "hybrid", "query_text": "What?",
        "metadata": {"query_id": "q1", "timestamp": "2026-01-01T00:00:00", "retrieval_method": "hybrid", "query_text": "What?"},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fixture, ensure_ascii=False) + "\n")
        f.write(json.dumps({**fixture, "query_id": "q2", "metadata": {**fixture["metadata"], "query_id": "q2"}}, ensure_ascii=False) + "\n")
    runs = get_session(sid)
    assert runs is not None
    assert len(runs) == 2
    assert runs[0].get("query_id") == "q1"
    assert runs[1].get("query_id") == "q2"


def test_export_csv_columns_and_citation_alignment():
    """Export CSV has required columns and every row chunk_id is in citation_mapping."""
    from prp.export import _package_to_evidence_rows
    pkg = {
        "query_id": "eq_01",
        "citation_mapping": [
            {"apa": "(Author et al., 2020)", "source_id": "src1", "chunk_id": "src1_c0001"},
            {"apa": "(Other et al., 2021)", "source_id": "src2", "chunk_id": "src2_c0002"},
        ],
        "retrieved_chunks": [
            {"chunk_id": "src1_c0001", "source_id": "src1", "text_preview": "First chunk preview."},
            {"chunk_id": "src2_c0002", "source_id": "src2", "text_preview": "Second chunk preview."},
        ],
    }
    rows = _package_to_evidence_rows(pkg, "eq_01")
    assert len(rows) == 2
    cols = ["query_id", "source_id", "chunk_id", "apa_citation", "evidence_snippet", "chunk_text_preview"]
    for r in rows:
        for c in cols:
            assert c in r
    chunk_ids = {r["chunk_id"] for r in rows}
    assert chunk_ids == {"src1_c0001", "src2_c0002"}


def test_export_session_to_csv_and_json(tmp_path):
    """Export session (fixture) to CSV and JSON; assert CSV has rows and JSON has required keys."""
    from prp.export import export_session_to_csv, export_session_to_json
    from prp.session import create_session, get_session
    from prp.config import SESSIONS_DIR
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid = create_session()
    path = SESSIONS_DIR / f"{sid}.jsonl"
    fixture = {
        "answer": "Answer.",
        "citations": [["s1", "s1_c0001"]],
        "citation_mapping": [{"apa": "(A, 2020)", "source_id": "s1", "chunk_id": "s1_c0001"}],
        "retrieved_chunks": [{"chunk_id": "s1_c0001", "source_id": "s1", "text_preview": "p"}],
        "query_id": "q1", "timestamp": "2026-01-01", "retrieval_method": "hybrid", "query_text": "Q",
        "metadata": {"query_id": "q1", "timestamp": "2026-01-01", "retrieval_method": "hybrid", "query_text": "Q"},
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(fixture, ensure_ascii=False) + "\n")
    csv_out = tmp_path / "out.csv"
    json_out = tmp_path / "out.json"
    export_session_to_csv(sid, csv_out)
    export_session_to_json(sid, json_out)
    assert csv_out.exists()
    with open(csv_out, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 1
    assert "chunk_id" in rows[0] and "apa_citation" in rows[0]
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert "answer" in data[0] and "citation_mapping" in data[0] and "retrieved_chunks" in data[0]


def test_manifest_validate_autofill():
    """manifest_validate autofill adds unknown_reason for missing required fields."""
    from prp.manifest_validate import autofill_missing_fields, _load_schema
    schema = _load_schema()
    entry = {"source_id": "s1", "title": "T"}
    filled = autofill_missing_fields(entry, schema)
    assert "unknown_reason" in filled
    assert "doc_id" in filled or filled.get("unknown_reason", "")


def test_manifest_validate_report():
    """validate_manifest returns report with entries, missing_per_field, valid."""
    from prp.manifest_validate import validate_manifest
    from prp.config import MANIFEST_PATH
    if not MANIFEST_PATH.exists():
        pytest.skip("No manifest.csv")
    report = validate_manifest(MANIFEST_PATH)
    assert "valid" in report
    assert "entries" in report
    assert "missing_per_field" in report


def test_evidence_table_build_and_render():
    """build_evidence_table does not crash; render_evidence_table_md/csv return non-empty strings."""
    from prp.artifacts import build_evidence_table, render_evidence_table_md, render_evidence_table_csv
    answer_text = "First claim is supported. Second claim is also supported by the literature."
    citations = [("src1", "src1_c0001")]
    retrieved_chunks = [
        {"chunk_id": "src1_c0001", "source_id": "src1", "text": "The literature supports this claim.", "rrf_score": 0.5},
    ]
    rows = build_evidence_table(answer_text, citations, retrieved_chunks)
    assert isinstance(rows, list)
    md = render_evidence_table_md(rows)
    assert "|" in md and "Claim" in md
    csv_str = render_evidence_table_csv(rows)
    assert "Claim" in csv_str and "Evidence snippet" in csv_str
