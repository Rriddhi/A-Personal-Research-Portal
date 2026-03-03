"""
Phase 3: Thread/session export — Markdown, JSONL, and ZIP bundle.
User-facing export for runs and full session.
"""
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

from .config import SESSIONS_DIR, ensure_dirs


def run_to_markdown(run: dict) -> str:
    """
    Markdown: Question, Answer, Citations, Evidence table summary, Retrieved chunks, Metadata.
    """
    lines = []
    title = run.get("query_text") or run.get("query") or "Run"
    lines.append(f"# {title[:200]}\n")
    lines.append("## Question\n")
    lines.append((run.get("query_text") or run.get("query") or "").strip())
    lines.append("\n## Answer\n")
    lines.append((run.get("answer") or "").strip())
    lines.append("\n## Citations\n")
    mapping = run.get("citation_mapping") or []
    for m in mapping:
        apa = m.get("apa") or ""
        cid = m.get("chunk_id") or ""
        lines.append(f"- {apa} → `{cid}`")
    if not mapping:
        lines.append("- (none)")
    lines.append("\n## Evidence table summary\n")
    chunks = run.get("retrieved_chunks") or []
    for i, c in enumerate(chunks[:20], 1):
        cid = c.get("chunk_id") or ""
        sid = c.get("source_id") or ""
        preview = (c.get("text") or c.get("text_preview") or "")[:400]
        lines.append(f"### Chunk {i}: {sid} / {cid}\n")
        lines.append(preview + ("..." if len(c.get("text") or c.get("text_preview") or "") > 400 else ""))
        lines.append("")
    lines.append("## Metadata\n")
    meta = run.get("metadata") or {}
    lines.append(f"- run_id: {run.get('run_id', '')}")
    lines.append(f"- timestamp: {run.get('timestamp', '')}")
    lines.append(f"- mode: {run.get('mode', '')}")
    lines.append(f"- query_id: {meta.get('query_id', '')}")
    lines.append(f"- retrieval_method: {meta.get('retrieval_method', '')}")
    return "\n".join(lines)


def session_to_jsonl_bytes(session_path: str | Path) -> bytes:
    """Return JSONL bytes exactly as stored. session_path can be path string or session_id."""
    path = Path(session_path)
    if not path.is_absolute():
        ensure_dirs()
        path = SESSIONS_DIR / f"{session_path}.jsonl" if not str(session_path).endswith(".jsonl") else SESSIONS_DIR / session_path
    if not path.exists():
        return b""
    return path.read_bytes()


def session_to_zip_bytes(
    session_id: str,
    runs: List[dict],
    include_pdf: bool = True,
) -> bytes:
    """
    ZIP with:
      - session.jsonl
      - runs/<run_id>.json
      - runs/<run_id>.md
      - runs/<run_id>.pdf (optional)
    """
    ensure_dirs()
    buf = BytesIO()
    session_path = SESSIONS_DIR / f"{session_id}.jsonl"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if session_path.exists():
            zf.writestr("session.jsonl", session_path.read_bytes())
        for r in runs:
            run_id = (r.get("run_id") or "run").replace(":", "-").replace(" ", "_")
            prefix = f"runs/{run_id}"
            zf.writestr(f"{prefix}.json", json.dumps(r, indent=2, ensure_ascii=False))
            zf.writestr(f"{prefix}.md", run_to_markdown(r))
            if include_pdf:
                try:
                    from .pdf_export import run_to_pdf
                    pdf_bytes = run_to_pdf(r, title=f"Run {run_id}")
                    zf.writestr(f"{prefix}.pdf", pdf_bytes)
                except Exception:
                    pass
    buf.seek(0)
    return buf.getvalue()
