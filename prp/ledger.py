"""
Source Selection Ledger: per-run record of source selection method, inclusion/exclusion, counts, sampling.
Saved under outputs/ledgers/{session_id}/{run_id}_source_selection.md
"""
from pathlib import Path
from typing import Any

from .config import OUTPUTS_LEDGERS_DIR, ensure_dirs


def write_source_selection_ledger(
    run_context: dict,
    manifest_stats: dict,
    session_id: str,
    run_id: str,
    selection_method: str = "scripted",
) -> str:
    """
    Write source selection ledger. Returns path to written file.
    run_context: dict with retrieval_config, retrieved (list of {source_id, chunk_id, score, method}), etc.
    manifest_stats: e.g. from manifest_validate.data_health_stats() or custom {n_docs, n_chunks, avg_chunk_len, n_missing_critical}.
    """
    ensure_dirs()
    base = OUTPUTS_LEDGERS_DIR / session_id
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{run_id}_source_selection.md"

    retrieval_config = run_context.get("retrieval_config") or {}
    method = retrieval_config.get("method", "hybrid_bm25_faiss_rrf")
    k = retrieval_config.get("k", 5)
    guardrails = retrieval_config.get("guardrails_enabled", True)

    retrieved = run_context.get("retrieved") or run_context.get("retrieved_chunks") or []
    n_chunks = len(retrieved)
    source_ids = list(dict.fromkeys(r.get("source_id") for r in retrieved if isinstance(r, dict) and r.get("source_id")))
    n_docs = len([s for s in source_ids if s])

    n_docs_manifest = manifest_stats.get("total_entries", manifest_stats.get("n_docs", 0))
    n_missing = manifest_stats.get("missing_per_field", {})
    if isinstance(n_missing, dict):
        n_missing_critical = sum(n_missing.get(f, 0) for f in ["doc_id", "title", "authors", "year"])
    else:
        n_missing_critical = manifest_stats.get("n_missing_critical", 0)

    chunk_lens = []
    for r in retrieved:
        if isinstance(r, dict):
            t = r.get("text") or r.get("text_preview") or ""
            chunk_lens.append(len(t))
    avg_chunk_len = sum(chunk_lens) / len(chunk_lens) if chunk_lens else 0

    lines = [
        "# Source Selection Ledger",
        "",
        f"**Run ID:** {run_id}",
        f"**Session ID:** {session_id}",
        "",
        "## Selection method",
        f"- **Method:** {selection_method} (scripted / manual / agentic)",
        f"- **Retrieval pipeline:** {method}",
        f"- **Top-k:** {k}",
        f"- **Guardrails enabled:** {guardrails}",
        "",
        "## Inclusion / Exclusion",
        "- **Inclusion:** Chunks from indexed corpus (FAISS + BM25); bibliography/boilerplate filtered; per-source cap applied.",
        "- **Exclusion:** Chunks below relevance threshold or bibliography score above threshold; optional topic-mismatch filter.",
        "",
        "## Counts",
        f"- **# docs (sources) in run:** {n_docs}",
        f"- **# chunks retrieved:** {n_chunks}",
        f"- **Avg chunk length (chars):** {avg_chunk_len:.0f}",
        f"- **# missing critical fields (manifest):** {n_missing_critical}",
        f"- **Manifest total entries:** {n_docs_manifest}",
        "",
        "## Sampling",
        "Top-k retrieval per query; no time filter. Diversification: max N chunks per source_id, then fill by RRF score.",
        "",
        "## Deterministic notes",
        "Pipeline versions: Phase 2 RAG (hybrid BM25 + FAISS, RRF, two-pass for limitations). Index built from data/processed/chunks.jsonl.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
