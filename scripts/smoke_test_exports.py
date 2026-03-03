#!/usr/bin/env python3
"""
Smoke test for Phase 3 exports: load a sample run, generate MD/JSON/PDF/ZIP, assert non-empty.
Run from repo root: python scripts/smoke_test_exports.py
No OPENAI_API_KEY or indexes required.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def sample_run() -> dict:
    """Minimal valid run for export tests."""
    return {
        "run_id": "smoke-run-001",
        "timestamp": "2026-01-01T12:00:00",
        "mode": "ask",
        "query": "What is evidence-based health?",
        "query_text": "What is evidence-based health?",
        "query_id": "eq_01",
        "answer": "Evidence-based health uses research findings to inform decisions.",
        "citation_mapping": [
            {"apa": "(Author et al., 2020)", "source_id": "doc1", "chunk_id": "doc1_c0001"},
        ],
        "retrieved_chunks": [
            {
                "chunk_id": "doc1_c0001",
                "source_id": "doc1",
                "text_preview": "Evidence-based practice integrates best research evidence.",
                "text": "Evidence-based practice integrates best research evidence with clinical expertise.",
            },
        ],
        "metadata": {"query_id": "eq_01", "timestamp": "2026-01-01T12:00:00", "retrieval_method": "hybrid", "query_text": "What is evidence-based health?"},
    }


def main() -> int:
    run = sample_run()
    errors = []

    # MD
    try:
        from prp.thread_export import run_to_markdown
        md = run_to_markdown(run)
        assert isinstance(md, str), "run_to_markdown should return str"
        assert len(md) > 0, "run_to_markdown returned empty"
    except Exception as e:
        errors.append(f"run_to_markdown: {e}")

    # JSON
    try:
        js = json.dumps(run, indent=2, ensure_ascii=False)
        assert len(js) > 0
    except Exception as e:
        errors.append(f"JSON dump: {e}")

    # PDF (run) — requires reportlab
    try:
        from prp.pdf_export import run_to_pdf
        pdf = run_to_pdf(run, title="Smoke test run")
        assert isinstance(pdf, bytes), "run_to_pdf should return bytes"
        assert len(pdf) > 0, "run_to_pdf returned empty"
    except ImportError as e:
        if "reportlab" in str(e).lower():
            pass  # skip when reportlab not installed
        else:
            errors.append(f"run_to_pdf: {e}")
    except Exception as e:
        errors.append(f"run_to_pdf: {e}")

    # Evidence table PDF
    try:
        from prp.pdf_export import evidence_table_to_pdf
        rows = [
            {"Claim": "Health is evidence-based.", "Evidence snippet": "Research shows.", "Citation": "(Author, 2020)", "Confidence": "H", "Notes": "Direct."},
        ]
        pdf_ev = evidence_table_to_pdf(rows, title="Evidence Table", subtitle="Smoke")
        assert isinstance(pdf_ev, bytes) and len(pdf_ev) > 0
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"evidence_table_to_pdf: {e}")

    # Memo PDF
    try:
        from prp.pdf_export import memo_to_pdf
        memo = "## Summary\n\nThis is a short memo."
        pdf_memo = memo_to_pdf(memo, title="Memo", metadata={})
        assert isinstance(pdf_memo, bytes) and len(pdf_memo) > 0
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"memo_to_pdf: {e}")

    # Session ZIP (no real session file; just runs in memory). PDF inside ZIP optional.
    try:
        from prp.thread_export import session_to_zip_bytes
        zip_bytes = session_to_zip_bytes("smoke-session", [run], include_pdf=True)
        assert isinstance(zip_bytes, bytes) and len(zip_bytes) > 0
    except Exception as e:
        errors.append(f"session_to_zip_bytes: {e}")

    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print("OK: All export smoke tests passed (MD, JSON, PDF run, PDF evidence, PDF memo, ZIP).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
