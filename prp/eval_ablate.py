"""
Ablation suite: BM25-only, FAISS-only, RRF-hybrid, RRF+Guardrails.
Runs eval per condition, writes outputs/eval/{condition}_runs.jsonl and {condition}_metrics.json.
Computes: retrieval_diversity, citation_coverage, unsupported_claim_rate, faithfulness_proxy.
"""
import json
import re
from pathlib import Path
from typing import Callable, List

from .config import (
    EVAL_QUERIES_JSON,
    OUTPUTS_EVAL_DIR,
    TOP_K,
    ensure_dirs,
)
from .retrieve import retrieve_bm25, retrieve_vector, retrieve_hybrid, retrieve_two_pass
from .generate import generate_answer
from .embed import embed_texts
from .evaluate import (
    load_eval_queries,
    compute_context_precision,
    compute_citation_precision,
)
from .utils import logger

INTERNAL_CITATION_PATTERN = re.compile(r"\(([A-Za-z0-9_\-\.]+(?:_[cC]\d+)?)\)")


def _citation_coverage(answer: str, citations: List[tuple]) -> float:
    """Fraction of answer sentences that have a citation (Evidence section citations / max(sentences in Answer, 1))."""
    idx_a = answer.find("Answer:")
    idx_e = answer.find("Evidence:")
    if idx_a < 0 or idx_e <= idx_a:
        return 0.0
    answer_block = answer[idx_a + 7 : idx_e]
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer_block) if s.strip() and len(s.strip()) > 20]
    n_sent = max(len(sentences), 1)
    cited = set()
    if citations:
        for c in citations:
            if len(c) >= 2 and c[1]:
                cited.add(c[1])
    evidence_section = answer[idx_e:answer.find("References:")] if "References:" in answer else answer[idx_e:]
    for m in INTERNAL_CITATION_PATTERN.finditer(evidence_section):
        cited.add(m.group(1))
    return min(1.0, len(cited) / n_sent)


def _unsupported_claim_rate(answer: str, citations: List[tuple]) -> float:
    """Proxy: 1 - citation_coverage (fraction of content not backed by a citation)."""
    return 1.0 - _citation_coverage(answer, citations)


def _faithfulness_proxy_mean(answer: str, citations: List[tuple], retrieved_chunks: List[dict]) -> float:
    """Mean cosine similarity between cited chunk text and answer (or cited sentence)."""
    if not citations or not retrieved_chunks:
        return 0.0
    chunk_by_id = {c.get("chunk_id", ""): c for c in retrieved_chunks if c.get("chunk_id")}
    answer_clean = answer[:2000]
    try:
        texts = [answer_clean]
        for sid, cid in citations:
            if not cid:
                continue
            c = chunk_by_id.get(cid, {})
            t = (c.get("text") or "")[:500]
            if t:
                texts.append(t)
        if len(texts) < 2:
            return 0.0
        vecs = embed_texts(texts)
        if not vecs or len(vecs) < 2:
            return 0.0
        q = vecs[0]
        sims = []
        for i in range(1, len(vecs)):
            a, b = q, vecs[i]
            dot = sum(x * y for x, y in zip(a, b))
            na = (sum(x * x for x in a)) ** 0.5
            nb = (sum(x * x for x in b)) ** 0.5
            if na > 0 and nb > 0:
                sims.append(dot / (na * nb))
        return float(sum(sims) / len(sims)) if sims else 0.0
    except Exception as e:
        logger.debug("Faithfulness proxy failed: %s", e)
        return 0.0


def _retrieval_diversity(retrieved_chunks: List[dict], top_k: int) -> int:
    """Number of distinct source_id in top-k."""
    if not retrieved_chunks:
        return 0
    sources = set()
    for c in retrieved_chunks[:top_k]:
        sid = c.get("source_id")
        if sid:
            sources.add(sid)
    return len(sources)


def _run_condition(
    condition: str,
    run_fn: Callable[[str, str], dict],
    queries_path: Path | None = None,
    max_queries: int | None = None,
) -> tuple[List[dict], dict]:
    """Run all queries with run_fn, compute ablation metrics. Returns (per_query_list, aggregate_dict). max_queries: cap for quick runs."""
    ensure_dirs()
    OUTPUTS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_eval_queries(queries_path)
    if max_queries is not None and max_queries > 0:
        queries = queries[:max_queries]
    per_query: List[dict] = []
    for q in queries:
        qid = q.get("query_id", "")
        qtext = q.get("query_text", "")
        try:
            result = run_fn(qtext, qid)
        except Exception as e:
            logger.warning("Condition %s query %s failed: %s", condition, qid, e)
            per_query.append({
                "query_id": qid,
                "context_precision": 0.0,
                "citation_precision": 0.0,
                "retrieval_diversity": 0,
                "citation_coverage": 0.0,
                "unsupported_claim_rate": 1.0,
                "faithfulness_proxy_mean": 0.0,
            })
            continue
        retrieved = result.get("retrieved_chunks", [])
        citations = result.get("citations", [])
        answer = result.get("answer", "")
        ctx_prec = compute_context_precision(retrieved, qtext)
        cit_prec = compute_citation_precision(answer, citations, retrieved)
        div = _retrieval_diversity(retrieved, TOP_K)
        cov = _citation_coverage(answer, citations)
        unsup = _unsupported_claim_rate(answer, citations)
        faith = _faithfulness_proxy_mean(answer, citations, retrieved)
        per_query.append({
            "query_id": qid,
            "context_precision": round(ctx_prec, 4),
            "citation_precision": round(cit_prec, 4),
            "retrieval_diversity": div,
            "citation_coverage": round(cov, 4),
            "unsupported_claim_rate": round(unsup, 4),
            "faithfulness_proxy_mean": round(faith, 4),
        })
    n = len(per_query)
    agg = {}
    for key in ["context_precision", "citation_precision", "citation_coverage", "unsupported_claim_rate", "faithfulness_proxy_mean"]:
        vals = [r[key] for r in per_query]
        agg[f"avg_{key}"] = round(sum(vals) / n, 4) if n else 0.0
    div_vals = [r["retrieval_diversity"] for r in per_query]
    agg["avg_retrieval_diversity"] = round(sum(div_vals) / n, 4) if n else 0.0
    return per_query, agg


def _make_bm25_run(top_k: int = TOP_K):
    def run(qtext: str, qid: str):
        chunks = retrieve_bm25(qtext, top_k=top_k)
        result = generate_answer(qtext, chunks)
        result["retrieved_chunks"] = chunks
        return result
    return run


def _make_faiss_run(top_k: int = TOP_K):
    def run(qtext: str, qid: str):
        chunks = retrieve_vector(qtext, top_k=top_k)
        result = generate_answer(qtext, chunks)
        result["retrieved_chunks"] = chunks
        return result
    return run


def _make_rrf_run(top_k: int = TOP_K):
    def run(qtext: str, qid: str):
        chunks = retrieve_hybrid(qtext, top_k=top_k)
        result = generate_answer(qtext, chunks)
        result["retrieved_chunks"] = chunks
        return result
    return run


def _make_rrf_guardrails_run(top_k: int = TOP_K):
    def run(qtext: str, qid: str):
        chunks = retrieve_two_pass(qtext, top_k=top_k)
        result = generate_answer(qtext, chunks)
        result["retrieved_chunks"] = chunks
        return result
    return run


CONDITIONS = {
    "bm25": _make_bm25_run,
    "faiss": _make_faiss_run,
    "rrf": _make_rrf_run,
    "rrf_guardrails": _make_rrf_guardrails_run,
}


def run_ablations(queries_path: Path | None = None, max_queries: int | None = None) -> dict:
    """
    Run all conditions, write outputs/eval/{condition}_runs.jsonl and {condition}_metrics.json.
    Returns {condition: {"per_query": [...], "aggregate": {...}}}. max_queries: cap for quick runs.
    """
    ensure_dirs()
    OUTPUTS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    queries_path = queries_path or EVAL_QUERIES_JSON
    results = {}
    for cond_name, run_factory in CONDITIONS.items():
        logger.info("Running ablation condition: %s", cond_name)
        per_query, agg = _run_condition(cond_name, run_factory(), queries_path, max_queries=max_queries)
        results[cond_name] = {"per_query": per_query, "aggregate": agg}
        runs_path = OUTPUTS_EVAL_DIR / f"{cond_name}_runs.jsonl"
        with open(runs_path, "w", encoding="utf-8") as f:
            for row in per_query:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_path = OUTPUTS_EVAL_DIR / f"{cond_name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    return results


if __name__ == "__main__":
    from .eval_view import compile_eval_summary_md
    run_ablations()
    compile_eval_summary_md()
    print("Done. See outputs/eval/")
