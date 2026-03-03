"""
Phase 3 quality gates: answer package schema validation and citation alignment.
Every citation must map to a retrieved chunk_id; no hallucinated citations.
"""
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = ("answer", "citations", "citation_mapping", "retrieved_chunks", "metadata")
METADATA_KEYS = ("query_id", "timestamp", "retrieval_method")


def _get_answer(pkg: dict) -> str:
    return (pkg.get("answer") or pkg.get("generated_answer") or "").strip()


def _get_citations(pkg: dict) -> list:
    c = pkg.get("citations")
    if c is None:
        return []
    return list(c)


def _get_citation_mapping(pkg: dict) -> list:
    m = pkg.get("citation_mapping")
    if m is None:
        return []
    return list(m)


def _get_retrieved_chunks(pkg: dict) -> list:
    r = pkg.get("retrieved_chunks")
    if r is None:
        return []
    return list(r)


def _get_metadata(pkg: dict) -> dict:
    if pkg.get("metadata") is not None:
        return dict(pkg["metadata"])
    return {
        "query_id": pkg.get("query_id", ""),
        "timestamp": pkg.get("timestamp", ""),
        "retrieval_method": pkg.get("retrieval_method", ""),
        "query_text": pkg.get("query_text", ""),
    }


def validate_answer_package(pkg: dict) -> tuple[bool, list[str]]:
    """
    Validate that the answer package has required keys and types.
    Accepts run_query-style dict (answer, citations, citation_mapping, retrieved_chunks)
    or session-line dict (generated_answer, citations, citation_mapping, retrieved_chunks, query_id, timestamp, ...).
    Returns (is_valid, list of error messages).
    """
    errors = []
    if not pkg or not isinstance(pkg, dict):
        return False, ["Package must be a non-empty dict"]

    # Required keys (allow generated_answer as alias for answer)
    answer = _get_answer(pkg)
    citations = _get_citations(pkg)
    citation_mapping = _get_citation_mapping(pkg)
    retrieved_chunks = _get_retrieved_chunks(pkg)
    metadata = _get_metadata(pkg)

    if not isinstance(answer, str):
        errors.append("'answer' must be a string")
    if not isinstance(citations, list):
        errors.append("'citations' must be a list")
    if not isinstance(citation_mapping, list):
        errors.append("'citation_mapping' must be a list")
    if not isinstance(retrieved_chunks, list):
        errors.append("'retrieved_chunks' must be a list")
    if not isinstance(metadata, dict):
        errors.append("'metadata' must be a dict")

    for key in METADATA_KEYS:
        if key not in metadata:
            errors.append(f"metadata must contain '{key}'")

    return len(errors) == 0, errors


def validate_citations_in_retrieved(pkg: dict) -> tuple[bool, list[str]]:
    """
    Every citation must correspond to a chunk_id in retrieved_chunks.
    citations are list of (source_id, chunk_id) or [source_id, chunk_id].
    citation_mapping is list of {apa, source_id, chunk_id}.
    Returns (is_valid, list of error messages).
    """
    errors = []
    retrieved_chunks = _get_retrieved_chunks(pkg)
    citation_mapping = _get_citation_mapping(pkg)
    citations = _get_citations(pkg)

    retrieved_ids = set()
    for c in retrieved_chunks:
        if isinstance(c, dict) and c.get("chunk_id"):
            retrieved_ids.add(c["chunk_id"])
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            # (source_id, chunk_id) or [source_id, chunk_id]
            retrieved_ids.add(c[1] if len(c) > 1 else c[0])

    for m in citation_mapping:
        if not isinstance(m, dict):
            continue
        cid = m.get("chunk_id")
        if not cid:
            continue
        if cid not in retrieved_ids:
            errors.append(f"Citation chunk_id '{cid}' not in retrieved_chunks")

    for cit in citations:
        if isinstance(cit, (list, tuple)) and len(cit) >= 2:
            cid = cit[1]
        elif isinstance(cit, dict):
            cid = cit.get("chunk_id")
        else:
            continue
        if cid and cid not in retrieved_ids:
            errors.append(f"Citation chunk_id '{cid}' not in retrieved_chunks")

    return len(errors) == 0, errors


def validate_package(pkg: dict) -> tuple[bool, list[str]]:
    """
    Run schema and citation validation. Returns (is_valid, all_errors).
    """
    ok, errs = validate_answer_package(pkg)
    if not ok:
        return False, errs
    ok2, errs2 = validate_citations_in_retrieved(pkg)
    return ok2, errs2
