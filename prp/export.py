"""
Phase 3: Export session (or single run) to CSV evidence table or JSON full run.
"""
import csv
import json
from pathlib import Path
from typing import Optional

from .config import SESSIONS_DIR, ensure_dirs
from .session import get_session
from .quality import validate_package


def _package_to_evidence_rows(pkg: dict, query_id: str) -> list[dict]:
    """One row per cited chunk: query_id, source_id, chunk_id, apa_citation, evidence_snippet, chunk_text_preview."""
    citation_mapping = pkg.get("citation_mapping") or []
    retrieved_chunks = pkg.get("retrieved_chunks") or []
    chunk_by_id = {c.get("chunk_id", ""): c for c in retrieved_chunks if c.get("chunk_id")}
    rows = []
    for m in citation_mapping:
        cid = m.get("chunk_id", "")
        sid = m.get("source_id", "")
        apa = m.get("apa", "")
        chunk = chunk_by_id.get(cid, {})
        preview = chunk.get("text_preview", chunk.get("text", ""))[:500]
        # evidence_snippet: could parse from answer Evidence section; here we use a short placeholder or first part of chunk
        evidence_snippet = preview[:200] if preview else ""
        rows.append({
            "query_id": query_id,
            "source_id": sid,
            "chunk_id": cid,
            "apa_citation": apa,
            "evidence_snippet": evidence_snippet,
            "chunk_text_preview": preview,
        })
    return rows


def export_session_to_csv(session_id: str, out_path: Path) -> Path:
    """Write evidence table CSV for all runs in session. Validates each package first."""
    runs = get_session(session_id)
    if runs is None:
        raise FileNotFoundError(f"Session not found: {session_id}")
    rows = []
    for pkg in runs:
        valid, errs = validate_package(pkg)
        if not valid:
            raise ValueError(f"Session contains invalid package: {errs}")
        query_id = pkg.get("query_id") or (pkg.get("metadata") or {}).get("query_id", "")
        rows.extend(_package_to_evidence_rows(pkg, query_id))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write header only
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "source_id", "chunk_id", "apa_citation", "evidence_snippet", "chunk_text_preview"])
            w.writeheader()
    else:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "source_id", "chunk_id", "apa_citation", "evidence_snippet", "chunk_text_preview"])
            w.writeheader()
            w.writerows(rows)
    return out_path


def export_session_to_json(session_id: str, out_path: Path) -> Path:
    """Write full run(s) as JSON. One JSON array of run objects."""
    runs = get_session(session_id)
    if runs is None:
        raise FileNotFoundError(f"Session not found: {session_id}")
    for pkg in runs:
        valid, errs = validate_package(pkg)
        if not valid:
            raise ValueError(f"Session contains invalid package: {errs}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)
    return out_path


def export_cmd(session_id: str, fmt: str, out_path: str) -> None:
    """CLI: export --session <id> --format csv|json --out <path>."""
    if fmt.lower() == "csv":
        export_session_to_csv(session_id, Path(out_path))
        print(f"Exported CSV to {out_path}")
    elif fmt.lower() == "json":
        export_session_to_json(session_id, Path(out_path))
        print(f"Exported JSON to {out_path}")
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use csv or json.")
