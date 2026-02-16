"""Run pipeline: retrieve + generate, log to phase2_runs.jsonl."""
import os

from .config import TOP_K, PHASE2_RUNS_JSONL, RUNS_DIR, ONCOLOGY_TERMS
from .retrieve import retrieve_hybrid, retrieve_hybrid_multi, retrieve_two_pass, LITERAL_BOOST_TERMS
from .generate import generate_answer
from .query_decompose import is_compare_query, build_subqueries
from .utils import append_run_log, save_run_to_file, utc_timestamp, logger, clean_text, preview_from_clean


def _apply_enhancement(chunks: list, query: str, top_k: int) -> list:
    """
    Enhanced RAG: expand retrieval (top_k*2) then rerank to top_k.
    The retrieve module already applies _rerank_by_relevance; we retrieve more here.
    """
    if len(chunks) <= top_k:
        return chunks
    return chunks[:top_k]


def _filter_off_topic_chunks(chunks: list, query: str, top_k: int) -> list:
    """
    Remove chunks with off-topic terms (e.g. cancer, chemotherapy) unless query mentions them.
    Soft fallback: if filtering would leave < top_k/2, keep all.
    """
    if not chunks:
        return chunks
    query_lower = query.lower()
    if any(t in query_lower for t in ONCOLOGY_TERMS):
        return chunks
    filtered = []
    for c in chunks:
        text = (c.get("text") or "").lower()
        if not any(t in text for t in ONCOLOGY_TERMS):
            filtered.append(c)
    if len(filtered) >= max(1, top_k // 2):
        return filtered
    return chunks

RETRIEVAL_METHOD = "hybrid_bm25_faiss_rrf"

TEXT_PREVIEW_LEN = 200


def run_query(
    query_text: str,
    query_id: str | None = None,
    top_k: int = TOP_K,
    mode: str = "baseline",
) -> dict:
    """
    Run retrieval + generation. mode: "llm_only" | "baseline" | "enhanced"
    - llm_only: no retrieval, LLM generates from no/minimal context
    - baseline: hybrid BM25+FAISS RRF, two-pass, extractive/LLM generation
    - enhanced: same as baseline but with rerank OR query rewrite (rerank via relevance)
    Returns dict with answer, citations, retrieved_chunks, and full log record.
    """
    # Show References + APA→chunk mapping by default for each query (override with SHOW_REFERENCES=0)
    os.environ.setdefault("SHOW_REFERENCES", "1")
    if query_id is None:
        query_id = f"q_{utc_timestamp().replace(':', '-')[:19]}"
    retrieval_method = RETRIEVAL_METHOD
    subqueries = None
    use_llm = os.environ.get("USE_LLM", "1").strip().lower() in ("1", "true", "yes")
    if mode == "llm_only":
        chunks = []
        retrieval_method = "llm_only"
        result = generate_answer(query_text, chunks, use_llm=True)
    else:
        eff_k = top_k * 2 if mode == "enhanced" else top_k
        subqueries = build_subqueries(query_text) if is_compare_query(query_text) else None
        if subqueries:
            chunks = retrieve_hybrid_multi(subqueries, top_k=eff_k)
        else:
            chunks = retrieve_two_pass(query_text, top_k=eff_k)
        chunks = _filter_off_topic_chunks(chunks, query_text, top_k)
        if mode == "enhanced":
            chunks = _apply_enhancement(chunks, query_text, top_k)
            retrieval_method = "hybrid_bm25_faiss_rrf_rerank"
        result = generate_answer(query_text, chunks, use_llm=use_llm)
    retrieved_for_log = [
        {
            "chunk_id": c["chunk_id"],
            "source_id": c["source_id"],
            "score": c.get("rrf_score"),
            "text_preview": preview_from_clean(clean_text(c.get("text") or ""), TEXT_PREVIEW_LEN),
        }
        for c in chunks
    ]
    log_record = {
        "timestamp": utc_timestamp(),
        "query_id": query_id,
        "query_text": query_text,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "retrieved_chunks": retrieved_for_log,
        "generated_answer": result["answer"],
        "citations": [list(c) for c in result["citations"]],
        "citation_mapping": result.get("citation_mapping", []),
        "model_name": result["model_name"],
        "prompt_version": result["prompt_version"],
    }
    if subqueries is not None:
        log_record["subqueries"] = subqueries
    append_run_log(log_record)
    save_run_to_file(log_record)

    # Debug: whether condition terms were found lexically in retrieved chunks
    terms_in_query = [t for t in LITERAL_BOOST_TERMS if t in (query_text or "").lower()]
    combined = " ".join((c.get("text") or "").lower() for c in chunks)
    condition_found_lexically = bool(
        terms_in_query and any(t in combined for t in terms_in_query)
    )

    return {
        "query_id": query_id,
        "answer": result["answer"],
        "citations": result["citations"],
        "citation_mapping": result.get("citation_mapping", []),
        "retrieved_chunks": chunks,
        "log_record": log_record,
        "retrieval_method": RETRIEVAL_METHOD,
        "top_k": top_k,
        "condition_found_lexically": condition_found_lexically,
    }


def run_demo() -> None:
    """One-command demo: build index if needed, run one query, print answer + citations, append log."""
    from .config import (
        CHUNKS_JSONL,
        FAISS_INDEX_PATH,
        DOCSTORE_PATH,
        BM25_DIR,
        ensure_dirs,
    )
    from .ingest import ingest_corpus
    from .chunk import run_chunking
    from .index import build_all_indexes

    ensure_dirs()
    if not CHUNKS_JSONL.exists():
        logger.info("Chunks not found; running ingestion + chunking...")
        ingest_corpus()
        run_chunking()
    if not CHUNKS_JSONL.exists() or CHUNKS_JSONL.stat().st_size == 0:
        raise RuntimeError(
            "No chunks produced. Ensure PyMuPDF is installed (pip install PyMuPDF) and "
            "data/raw/ contains PDFs. Use the same Python for pip and for running this command."
        )
    if not FAISS_INDEX_PATH.exists() or not DOCSTORE_PATH.exists() or not (BM25_DIR / "bm25.pkl").exists():
        logger.info("Index not found; building FAISS + BM25...")
        build_all_indexes()
    # Demo shows References + APA-to-Chunk Mapping by default (override with SHOW_REFERENCES=0)
    os.environ.setdefault("SHOW_REFERENCES", "1")
    demo_query = "What evidence exists for personalized health or dietary recommendations in the corpus?"
    result = run_query(demo_query, query_id="phase2_demo")
    print("--- Retrieved chunk IDs ---")
    for c in result["retrieved_chunks"]:
        print(c["chunk_id"], c["source_id"])
    print("\n--- Answer (synthesized + Evidence + References) ---")
    print(result["answer"])
    print("\n--- Citations (source_id, chunk_id) ---")
    for cit in result["citations"]:
        print(cit)
    mapping = result.get("citation_mapping", [])
    if mapping:
        print("\n--- APA → Evidence Chunk Mapping ---")
        for m in mapping:
            print(f"  {m.get('apa', '')} → {m.get('chunk_id', '')}")
    print("\n--- Log appended to", PHASE2_RUNS_JSONL, "---")
