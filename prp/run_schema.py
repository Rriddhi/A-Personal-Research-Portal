"""
Lightweight source of truth for Run object schema (session run record).
Used by session, export, and UI. Backward-compatible with existing JSONL lines.
"""
from typing import List, TypedDict


class RunRetrievedChunk(TypedDict, total=False):
    """Single chunk in run. text_preview always; text optional (for citation explorer)."""
    chunk_id: str
    source_id: str
    text_preview: str
    text: str  # full chunk text when persisted (new runs)
    score: float
    page: str


class RunCitationMapping(TypedDict, total=False):
    """One citation entry in citation_mapping."""
    apa: str
    source_id: str
    chunk_id: str


class RunDict(TypedDict, total=False):
    """Session run record (one line in logs/sessions/<session_id>.jsonl)."""
    run_id: str
    timestamp: str
    mode: str  # "ask" | "search" | "artifact" | "eval"
    query: str
    query_text: str
    query_id: str
    retrieval_config: dict
    model_config: dict
    retrieved: list
    retrieved_chunks: List[dict]
    answer: str
    citation_mapping: List[dict]
    citations: list
    diagnostics: dict
    artifacts: list
    ledger_path: str
    notes: str
    metadata: dict
