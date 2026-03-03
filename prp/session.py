"""
Phase 3: Session-based querying. Persistent history per session in logs/sessions/<session_id>.jsonl.

Structured run schema (per line):
  run_id, timestamp, mode (search|ask|artifact|eval), query/question,
  retrieval_config {method, k, weights, guardrails_enabled},
  model_config {model_name, temperature, prompt_version},
  retrieved [{source_id, chunk_id, score, method}],
  answer {text, citations: [{source_id, chunk_id, span_text}]},
  diagnostics {citation_coverage, unsupported_claim_rate, faithfulness_proxy_mean},
  artifacts [{type, path, format}], ledger_path, notes.
Legacy lines may omit run_id/mode; treated as mode "ask".
"""
import json
import uuid
from pathlib import Path
from typing import Any, Optional

from .config import SESSIONS_DIR, ensure_dirs
from .run import run_query
from .utils import utc_timestamp, logger, clean_text, preview_from_clean
from .quality import validate_package

TEXT_PREVIEW_LEN = 200


def _session_path(session_id: str) -> Path:
    ensure_dirs()
    return SESSIONS_DIR / f"{session_id}.jsonl"


def create_session() -> str:
    """Create a new session. Returns session_id (UUID)."""
    ensure_dirs()
    session_id = str(uuid.uuid4())
    path = _session_path(session_id)
    # Touch file with a header line (optional; first append will create anyway)
    path.write_text("", encoding="utf-8")
    logger.info("Created session %s", session_id)
    return session_id


def list_sessions() -> list[dict]:
    """List all sessions: [{ session_id, path, created_ts, num_queries }, ...]."""
    ensure_dirs()
    if not SESSIONS_DIR.exists():
        return []
    out = []
    for f in sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        session_id = f.stem
        lines = [ln for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        first_ts = ""
        if lines:
            try:
                first = json.loads(lines[0])
                first_ts = first.get("timestamp", first.get("metadata", {}).get("timestamp", ""))
            except Exception:
                pass
        out.append({
            "session_id": session_id,
            "path": str(f),
            "created_ts": first_ts or str(f.stat().st_mtime),
            "num_queries": len(lines),
        })
    return out


def get_session(session_id: str) -> Optional[list[dict]]:
    """Load session runs. Returns list of run records (one per line) or None if not found."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    runs = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs


def load_sessions() -> list[dict]:
    """List all sessions with session_id, path, created_ts, num_queries. Alias for list_sessions() result."""
    return list_sessions()


def load_session(session_id: str) -> Optional[list[dict]]:
    """Load runs for a session. Alias for get_session()."""
    return get_session(session_id)


def list_runs(session_id: str) -> list[dict]:
    """List run records for a session. Same as get_session() but returns [] if not found."""
    runs = get_session(session_id)
    return runs if runs is not None else []


def _build_package_for_storage(result: dict, log_record: dict) -> dict:
    """Build one JSON-serializable run record for session file. Fills structured run schema."""
    rec = dict(log_record)
    rec["answer"] = result.get("answer", "")
    rec["citation_mapping"] = [dict(m) for m in (result.get("citation_mapping") or log_record.get("citation_mapping") or [])]
    chunks = result.get("retrieved_chunks") or []
    rec["retrieved_chunks"] = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "source_id": c.get("source_id", ""),
            "text_preview": preview_from_clean(clean_text(c.get("text") or ""), TEXT_PREVIEW_LEN),
            "score": c.get("rrf_score") or c.get("vector_score") or c.get("bm25_score"),
            # Persist full text for citation explorer (optional; old runs may lack it)
            **({"text": clean_text(c.get("text") or "")} if c.get("text") else {}),
        }
        for c in chunks
    ]
    rec["metadata"] = {
        "query_id": rec.get("query_id", ""),
        "timestamp": rec.get("timestamp", ""),
        "retrieval_method": rec.get("retrieval_method", ""),
        "query_text": rec.get("query_text", ""),
    }
    # Structured run schema
    rec.setdefault("run_id", str(uuid.uuid4()))
    rec.setdefault("mode", "ask")
    rec.setdefault("retrieval_config", {
        "method": rec.get("retrieval_method", "hybrid_bm25_faiss_rrf"),
        "k": rec.get("top_k", 5),
        "guardrails_enabled": True,
    })
    rec.setdefault("model_config", {
        "model_name": result.get("model_name", ""),
        "prompt_version": result.get("prompt_version", ""),
    })
    rec.setdefault("retrieved", [
        {"source_id": c.get("source_id"), "chunk_id": c.get("chunk_id"), "score": c.get("score"), "method": rec.get("retrieval_method")}
        for c in rec["retrieved_chunks"]
    ])
    rec.setdefault("diagnostics", {})
    rec.setdefault("artifacts", [])
    rec.setdefault("ledger_path", "")
    rec.setdefault("notes", "")
    return rec


def append_run(session_id: str, run_obj: dict) -> None:
    """Append a structured run object to the session file. Does not validate package."""
    path = _session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_obj.setdefault("run_id", run_obj.get("run_id") or str(uuid.uuid4()))
    run_obj.setdefault("timestamp", utc_timestamp())
    run_obj.setdefault("mode", "ask")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_obj, ensure_ascii=False) + "\n")
    logger.info("Appended run %s to session %s", run_obj.get("run_id"), session_id)


def append_query_run(session_id: str, query_text: str, query_id: Optional[str] = None) -> dict:
    """
    Run query via run_query(), validate answer package, append to session file.
    Returns the same dict as run_query() (includes answer, citations, retrieved_chunks, log_record).
    """
    result = run_query(query_text, query_id=query_id)
    log_record = result.get("log_record", {})
    package = _build_package_for_storage(result, log_record)
    valid, errors = validate_package(package)
    if not valid:
        logger.warning("Phase 3 quality validation failed: %s", errors)
        raise ValueError(f"Answer package validation failed: {errors}")
    path = _session_path(session_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(package, ensure_ascii=False) + "\n")
    logger.info("Appended query to session %s", session_id)
    return result


def session_new_cmd() -> str:
    """CLI: create session, print session_id."""
    sid = create_session()
    print(sid)
    return sid


def session_list_cmd() -> None:
    """CLI: list sessions."""
    sessions = list_sessions()
    if not sessions:
        print("No sessions found.")
        return
    for s in sessions:
        print(f"{s['session_id']}\t{s['created_ts']}\t{s['num_queries']} queries")


def session_show_cmd(session_id: str) -> None:
    """CLI: show session summary and query history (no full chunk text)."""
    runs = get_session(session_id)
    if runs is None:
        print(f"Session not found: {session_id}")
        return
    print(f"Session: {session_id}")
    print(f"Queries: {len(runs)}")
    for i, r in enumerate(runs, 1):
        qid = r.get("query_id") or r.get("metadata", {}).get("query_id", "")
        ts = r.get("timestamp") or r.get("metadata", {}).get("timestamp", "")
        q = (r.get("query_text") or r.get("metadata", {}).get("query_text", ""))[:60]
        print(f"  {i}. [{ts}] {qid}  {q}...")


def session_query_cmd(session_id: str, query_text: str) -> dict:
    """CLI: run query in session, print answer, return run result."""
    result = append_query_run(session_id, query_text)
    print(result["answer"])
    return result
