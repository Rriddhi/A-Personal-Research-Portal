"""Retrieval: hybrid BM25 + vector (FAISS), merge with reciprocal rank fusion."""
import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import (
    INDEXES_DIR,
    FAISS_INDEX_PATH,
    DOCSTORE_PATH,
    BM25_DIR,
    TOP_K,
    RRF_K,
    MAX_PER_SOURCE,
    RELEVANCE_RRF_WEIGHT,
    RELEVANCE_SIM_WEIGHT,
    RELEVANCE_MIN_SIMILARITY,
    TOPIC_MISMATCH_GUARDRAIL,
    ONCOLOGY_TERMS,
    TOPIC_MISMATCH_PENALTY,
    BIBLIOGRAPHY_THRESHOLD,
    BIBLIOGRAPHY_THRESHOLD_RELAXED,
)
from .embed import embed_texts
from .utils import logger, is_boilerplate_with_bib_threshold, evidence_strength, evidence_type_label

# Truncate chunk text for embedding (avoid huge inputs)
CHUNK_TEXT_FOR_SIMILARITY = 2000

_faiss = None
_bm25 = None
_docstore = None
_chunk_ids = None


def _import_faiss():
    try:
        import faiss
        return faiss
    except ImportError as e:
        raise ImportError(
            "faiss is not installed in this environment. Install it with: pip install faiss-cpu"
        ) from e


def load_faiss_and_docstore() -> tuple:
    """Load FAISS index and docstore. Returns (index, docstore_list)."""
    global _faiss, _docstore
    if not FAISS_INDEX_PATH.exists() or not DOCSTORE_PATH.exists():
        raise FileNotFoundError(
            "Index files not found. Run: make ingest && make build_index"
        )
    faiss = _import_faiss()
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        _docstore = json.load(f)
    return index, _docstore


def load_bm25() -> tuple:
    """Load BM25 and chunk_ids from BM25_DIR. Returns (bm25, chunk_ids)."""
    global _bm25, _chunk_ids
    with open(BM25_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    _chunk_ids = meta["chunk_ids"]
    with open(BM25_DIR / "corpus.json", "r", encoding="utf-8") as f:
        tokenized = json.load(f)
    with open(BM25_DIR / "bm25.pkl", "rb") as f:
        _bm25 = pickle.load(f)
    return _bm25, _chunk_ids


def rrf_scores(rank_lists: List[List[str]], k: int = RRF_K) -> dict:
    """Reciprocal Rank Fusion: score(doc) = sum 1/(k + rank). Returns doc_id -> score."""
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 if norms are zero."""
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    dot = float(np.dot(a, b))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def _rerank_by_relevance(
    chunks: List[dict],
    query: str,
    top_k: int,
) -> List[dict]:
    """
    Re-rank chunks by weighted combination of RRF score + query–chunk embedding similarity.
    Topic mismatch guardrail: when query does not mention oncology, penalize or filter chunks
    containing oncology terms (cancer, chemotherapy, etc.). Prevents domain drift; improves relevance.
    Filter preferred when enough non-oncology chunks exist; otherwise apply penalty (soft fallback).
    """
    if not chunks or not query:
        return chunks[:top_k]
    query_lower = query.lower()
    # Embed query and chunk texts (truncate chunk for embedding)
    texts = [query] + [(c.get("text") or "")[:CHUNK_TEXT_FOR_SIMILARITY] for c in chunks]
    try:
        vecs = embed_texts(texts)
    except Exception as e:
        logger.debug("Relevance re-rank skipped (embed failed): %s", e)
        return chunks[:top_k]
    if not vecs or len(vecs) != len(texts):
        return chunks[:top_k]
    q_vec = np.array(vecs[0], dtype=np.float32)
    chunk_vecs = [np.array(v, dtype=np.float32) for v in vecs[1:]]
    rrf_max = max((c.get("rrf_score") or 0.0) for c in chunks) or 1.0
    scored = []
    for i, c in enumerate(chunks):
        rrf = c.get("rrf_score") or 0.0
        rrf_norm = rrf / rrf_max
        sim = _cosine_sim(q_vec, chunk_vecs[i]) if i < len(chunk_vecs) else 0.0
        combined = RELEVANCE_RRF_WEIGHT * rrf_norm + RELEVANCE_SIM_WEIGHT * max(0.0, sim)
        chunk_lower = (c.get("text") or "").lower()
        has_oncology = any(term in chunk_lower for term in ONCOLOGY_TERMS)
        query_has_oncology = any(term in query_lower for term in ONCOLOGY_TERMS)
        # Topic mismatch: penalize oncology chunks when query doesn't ask about oncology
        if TOPIC_MISMATCH_GUARDRAIL and not query_has_oncology and has_oncology:
            combined *= TOPIC_MISMATCH_PENALTY
        scored.append((combined, sim, has_oncology, c))
    # Sort by combined score descending
    scored.sort(key=lambda x: -x[0])
    # Prefer chunks above min similarity; if we have enough, drop below threshold
    above = [(s, sim, has_onc, c) for s, sim, has_onc, c in scored if sim >= RELEVANCE_MIN_SIMILARITY]
    non_oncology_above = [(s, sim, c) for s, sim, has_onc, c in above if not has_onc]
    non_oncology_scored = [(s, sim, c) for s, sim, has_onc, c in scored if not has_onc]
    # Filter oncology chunks from top_k when we have enough non-oncology alternatives (soft fallback)
    if TOPIC_MISMATCH_GUARDRAIL and not query_has_oncology:
        if len(non_oncology_above) >= top_k:
            out = [c for _, _, c in non_oncology_above[:top_k]]
        elif len(non_oncology_scored) >= top_k:
            out = [c for _, _, c in non_oncology_scored[:top_k]]
        else:
            out = [c for _, _, _, c in scored[:top_k]]
    else:
        if len(above) >= top_k:
            out = [c for _, _, _, c in above[:top_k]]
        else:
            out = [c for _, _, _, c in scored[:top_k]]
    return out


def retrieve_vector(query: str, top_k: int = TOP_K) -> List[dict]:
    """Retrieve top_k chunks using FAISS (embed query, search)."""
    index, docstore = load_faiss_and_docstore()
    q_vec = np.array(embed_texts([query]), dtype=np.float32)
    faiss = _import_faiss()
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, min(top_k * 2, len(docstore)))
    out = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(docstore):
            continue
        rec = docstore[idx]
        rec["vector_score"] = float(scores[0][i])
        out.append(rec)
        if len(out) >= top_k:
            break
    return out


def retrieve_bm25(query: str, top_k: int = TOP_K) -> List[dict]:
    """Retrieve top_k chunks using BM25."""
    bm25, chunk_ids = load_bm25()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    order = np.argsort(-scores)
    docstore_path = DOCSTORE_PATH
    with open(docstore_path, "r", encoding="utf-8") as f:
        docstore = json.load(f)
    id_to_doc = {d["chunk_id"]: d for d in docstore}
    out = []
    for idx in order:
        if scores[idx] <= 0:
            continue
        cid = chunk_ids[idx]
        doc = id_to_doc.get(cid, {"chunk_id": cid, "source_id": "", "text": ""})
        doc["bm25_score"] = float(scores[idx])
        out.append(doc)
        if len(out) >= top_k:
            break
    return out


def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
    max_per_source: int = MAX_PER_SOURCE,
) -> List[dict]:
    """Hybrid retrieval: BM25 + vector + literal keyword boost, merge with RRF, then diversify by capping per source_id."""
    vec_results = retrieve_vector(query, top_k=top_k * 2)
    bm25_results = retrieve_bm25(query, top_k=top_k * 2)
    literal_results = _retrieve_literal_keyword_chunks(query, top_k=top_k * 2)
    rank_lists = [
        [r["chunk_id"] for r in vec_results],
        [r["chunk_id"] for r in bm25_results],
    ]
    if literal_results:
        rank_lists.append([r["chunk_id"] for r in literal_results])
    scores = rrf_scores(rank_lists, k=rrf_k)
    candidates = vec_results + bm25_results
    if literal_results:
        # Add literal chunks to candidate pool (may overlap; _get_doc dedupes by cid)
        for r in literal_results:
            if not any(c["chunk_id"] == r["chunk_id"] for c in candidates):
                candidates.append(r)
    merged = _merged_chunks_from_rrf_scores(
        scores, candidates, top_k, max_per_source=max_per_source,
    )
    merged = _rerank_by_relevance(merged, query, top_k)
    logger.info(
        "Retrieved top_k=%d (max_per_source=%d): %s",
        len(merged),
        max_per_source,
        [(c["chunk_id"], c["source_id"]) for c in merged],
    )
    return merged


def _get_doc(cid: str, candidate_chunks: List[dict]) -> dict:
    """Resolve chunk_id to doc (chunk_id, source_id, text)."""
    doc = next((r for r in candidate_chunks if r["chunk_id"] == cid), None)
    if doc is not None:
        return doc
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore = json.load(f)
    return next((d for d in docstore if d["chunk_id"] == cid), {"chunk_id": cid, "source_id": "", "text": ""})


def _merged_chunks_from_rrf_scores(
    scores: dict,
    candidate_chunks: List[dict],
    top_k: int,
    max_per_source: int = MAX_PER_SOURCE,
) -> List[dict]:
    """
    Filter candidates BEFORE selecting top_k (boilerplate + bibliography), then apply
    source diversification. Fallback: if filtering removes too many, relax bibliography
    threshold and retry.
    """
    sorted_cids = sorted(scores.keys(), key=lambda x: -scores[x])

    def _select(bib_threshold: float) -> List[str]:
        selected: List[str] = []
        per_source: dict = {}
        filter_fn = lambda t: is_boilerplate_with_bib_threshold(t, bibliography_threshold=bib_threshold)

        # Phase 1: filter then take up to max_per_source per source until top_k
        for cid in sorted_cids:
            if len(selected) >= top_k:
                break
            doc = _get_doc(cid, candidate_chunks)
            if filter_fn(doc.get("text") or ""):
                continue
            sid = doc.get("source_id") or ""
            if per_source.get(sid, 0) >= max_per_source:
                continue
            selected.append(cid)
            per_source[sid] = per_source.get(sid, 0) + 1

        # Phase 2: fill remaining slots from remainder (no per-source cap)
        if len(selected) < top_k:
            selected_set = set(selected)
            for cid in sorted_cids:
                if cid in selected_set:
                    continue
                doc = _get_doc(cid, candidate_chunks)
                if filter_fn(doc.get("text") or ""):
                    continue
                selected.append(cid)
                selected_set.add(cid)
                if len(selected) >= top_k:
                    break
        return selected

    selected = _select(BIBLIOGRAPHY_THRESHOLD)
    if len(selected) < top_k:
        selected = _select(BIBLIOGRAPHY_THRESHOLD_RELAXED)
        if len(selected) > 0:
            logger.debug("Relaxed bibliography threshold to %s; selected %d chunks", BIBLIOGRAPHY_THRESHOLD_RELAXED, len(selected))

    merged = []
    for cid in selected:
        doc = _get_doc(cid, candidate_chunks)
        text = doc.get("text") or ""
        merged.append({
            "chunk_id": doc["chunk_id"],
            "source_id": doc["source_id"],
            "text": doc["text"],
            "rrf_score": scores[cid],
            "evidence_strength": evidence_strength(text),
            "evidence_type": evidence_type_label(text),
        })
    return merged


# Condition/entity and intervention terms: when in query, boost chunks containing them (literal match)
# so rare terms like "vitiligo" surface even if semantic/BM25 rank them low
LITERAL_BOOST_TERMS = (
    "vitiligo", "celery", "detox", "smoothie", "autoimmune", "skin",
    "heavy metal", "metal detox", "smoothies", "celery juice", "reverses",
)


def _retrieve_literal_keyword_chunks(query: str, top_k: int = 5) -> List[dict]:
    """
    Scan docstore for chunks containing any LITERAL_BOOST_TERMS that appear in the query.
    Returns chunks that mention those terms; used to surface rare keywords BM25/semantic may miss.
    """
    q_lower = (query or "").lower()
    terms_in_query = [t for t in LITERAL_BOOST_TERMS if t in q_lower]
    if not terms_in_query:
        return []
    if not DOCSTORE_PATH.exists():
        return []
    try:
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            docstore = json.load(f)
    except Exception:
        return []
    out = []
    seen = set()
    for d in docstore:
        text = (d.get("text") or "").lower()
        if not text:
            continue
        for term in terms_in_query:
            if term in text and d.get("chunk_id") not in seen:
                seen.add(d.get("chunk_id"))
                out.append({
                    "chunk_id": d.get("chunk_id", ""),
                    "source_id": d.get("source_id", ""),
                    "text": d.get("text", ""),
                    "literal_boost_score": 1.0,
                })
                break
        if len(out) >= top_k:
            break
    return out


# Suffix for limitations-focused retrieval (Pass2)
LIMITATIONS_QUERY_SUFFIX = (
    "\nlimitations OR uncertainty OR bias OR generalizability OR adherence OR "
    "long-term OR confounding OR trial limitations"
)


def retrieve_two_pass(
    query: str,
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
    max_per_source: int = MAX_PER_SOURCE,
) -> List[dict]:
    """
    Two-pass retrieval: Pass1 = answer-focused, Pass2 = limitations-focused.
    Merge with dedupe by chunk_id, keep per-source cap. Tag chunks with
    'retrieval_pass': 'answer' | 'limitations'.
    """
    pass1 = retrieve_hybrid(query, top_k=top_k * 2, rrf_k=rrf_k, max_per_source=max_per_source)
    for c in pass1:
        c["retrieval_pass"] = "answer"

    limitations_query = query.strip() + LIMITATIONS_QUERY_SUFFIX
    pass2 = retrieve_hybrid(limitations_query, top_k=top_k * 2, rrf_k=rrf_k, max_per_source=max_per_source)
    for c in pass2:
        c["retrieval_pass"] = "limitations"

    seen = {c["chunk_id"]: c for c in pass1}
    limitations_only = [c for c in pass2 if c["chunk_id"] not in seen]
    seen.update({c["chunk_id"]: c for c in limitations_only})

    # Build merged list: pass1 first, then pass2-only, respecting per-source cap
    merged: List[dict] = []
    per_source: dict = {}
    for c in pass1:
        if len(merged) >= top_k:
            break
        sid = c.get("source_id") or ""
        if per_source.get(sid, 0) >= max_per_source:
            continue
        merged.append(c)
        per_source[sid] = per_source.get(sid, 0) + 1

    for c in limitations_only:
        if len(merged) >= top_k * 2:
            break
        sid = c.get("source_id") or ""
        if per_source.get(sid, 0) >= max_per_source:
            continue
        merged.append(c)
        per_source[sid] = per_source.get(sid, 0) + 1

    pass1_ids = {c["chunk_id"] for c in pass1}
    limitations_ids = [c["chunk_id"] for c in merged if c.get("retrieval_pass") == "limitations"]
    pass1_chunks = [c for c in merged if c.get("retrieval_pass") == "answer"]
    logger.info("Retrieved Pass1 top_k=%d: %s", len(pass1_chunks), [c["chunk_id"] for c in pass1_chunks])
    logger.info("Retrieved Pass2 top_k=%d: %s", len(limitations_ids), limitations_ids)
    return merged


def retrieve_hybrid_multi(
    subqueries: List[str],
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
    max_per_source: int = MAX_PER_SOURCE,
) -> List[dict]:
    """
    Retrieve top_k for each subquery, merge with RRF, diversify by max_per_source, return top_k chunks.
    Used for compare/difference query decomposition.
    """
    if not subqueries:
        return []
    rank_lists = []
    all_chunks_by_cid = {}
    for q in subqueries:
        chunks = retrieve_hybrid(q, top_k=top_k * 2, rrf_k=rrf_k, max_per_source=max_per_source)
        rank_lists.append([c["chunk_id"] for c in chunks])
        for c in chunks:
            all_chunks_by_cid[c["chunk_id"]] = c
    scores = rrf_scores(rank_lists, k=rrf_k)
    candidates = list(all_chunks_by_cid.values())
    merged = _merged_chunks_from_rrf_scores(scores, candidates, top_k, max_per_source=max_per_source)
    # Re-rank by relevance using first subquery as effective query
    merged = _rerank_by_relevance(merged, subqueries[0] if subqueries else "", top_k)
    logger.info(
        "Retrieved (multi) top_k=%d (max_per_source=%d): %s",
        len(merged),
        max_per_source,
        [(c["chunk_id"], c["source_id"]) for c in merged],
    )
    return merged
