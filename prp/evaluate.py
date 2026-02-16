"""
Evaluation: Phase 2 matrix-aligned metrics.
Implements Context Precision, Context Recall, Faithfulness, Citation Precision,
Answer Relevance, Coherence, Conciseness, Artifact Readiness.
"""
import json
import csv
import re
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from .config import (
    EVAL_QUERIES_JSON,
    METRICS_DIR,
    EVAL_SUMMARY_JSON,
    EVAL_PER_QUERY_CSV,
    PER_QUERY_METRICS_JSON,
    AGGREGATE_METRICS_JSON,
    EVALUATION_SUMMARY_MD,
    ensure_dirs,
)
from .run import run_query
from .generate import parse_citations_from_answer
from .utils import logger, save_run_to_file

# Internal citation format: (source_id_chunkID) e.g. (jmir-2023-1-e37667_c0006)
INTERNAL_CITATION_PATTERN = re.compile(
    r"\(([A-Za-z0-9_\-\.]+(?:_[cC]\d+)?)\)"
)


def load_eval_queries(path: Path | None = None) -> List[dict]:
    path = path or EVAL_QUERIES_JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", data) if isinstance(data, dict) else data


def load_chunk_texts() -> Dict[str, str]:
    """Load chunk_id -> text from chunks.jsonl."""
    from .config import CHUNKS_JSONL
    chunk_texts = {}
    if not CHUNKS_JSONL.exists():
        return chunk_texts
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_texts[rec["chunk_id"]] = rec.get("text", "")
    return chunk_texts


# ---- Component-wise metrics ----

def _chunk_relevant_to_query(chunk_text: str, query: str) -> bool:
    """Keyword-based relevance: chunk relevant if it shares substantive terms with query."""
    q_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 3)
    if not q_words:
        return True
    text_lower = (chunk_text or "").lower()
    overlap = sum(1 for w in q_words if w in text_lower)
    return overlap >= max(1, len(q_words) * 0.15)


def compute_context_precision(retrieved_chunks: List[dict], query: str) -> float:
    """
    (# retrieved chunks relevant to query) / (total retrieved chunks)
    Uses keyword-based relevance when LLM not available.
    """
    if not retrieved_chunks:
        return 1.0
    relevant = sum(
        1 for c in retrieved_chunks
        if _chunk_relevant_to_query(c.get("text", ""), query)
    )
    return relevant / len(retrieved_chunks)


def compute_context_recall(
    retrieved_chunks: List[dict],
    query: str,
    total_relevant_in_corpus: Optional[int] = None,
) -> float:
    """
    (# relevant chunks retrieved) / (total relevant in corpus subset).
    Without labeled dev set: approximate via retrieved relevant count / (retrieved + estimate missed).
    Returns 0.5 as placeholder when total_relevant unknown.
    """
    if not retrieved_chunks:
        return 0.0
    relevant_retrieved = sum(
        1 for c in retrieved_chunks
        if _chunk_relevant_to_query(c.get("text", ""), query)
    )
    if total_relevant_in_corpus is not None and total_relevant_in_corpus > 0:
        return relevant_retrieved / total_relevant_in_corpus
    return relevant_retrieved / max(len(retrieved_chunks), 1)


def _llm_score_1_4(prompt: str, default: float = 2.5) -> float:
    """Call LLM for 1-4 rubric score. Returns default if no API key or error."""
    if not os.environ.get("OPENAI_API_KEY"):
        return default
    try:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        for part in content.replace(",", " ").split():
            try:
                v = float(part)
                if 1 <= v <= 4:
                    return v
            except ValueError:
                continue
        return default
    except Exception as e:
        logger.debug("LLM metric failed: %s", e)
        return default


def compute_faithfulness(answer: str, retrieved_texts: List[str]) -> float:
    """
    Are claims in answer supported by retrieved chunks? LLM score 1-4.
    """
    context = "\n\n".join((t or "")[:500] for t in retrieved_texts[:8])
    prompt = (
        f"Rate whether the claims in the answer are supported by the retrieved context.\n"
        f"Context:\n{context[:3000]}\n\nAnswer:\n{answer[:2000]}\n\n"
        "Score 1-4: 1=unsupported/hallucinated, 2=partially supported, 3=mostly supported, 4=fully supported. "
        "Reply with only the number."
    )
    return _llm_score_1_4(prompt, default=2.5)


# ---- End-to-end metrics ----

def parse_citations_from_evidence_section(full_output: str) -> List[str]:
    """
    Extract chunk_ids from Evidence section only (not Answer).
    Returns list of chunk_id strings found in Evidence.
    """
    idx_e = full_output.find("Evidence:")
    idx_r = full_output.find("References:")
    if idx_e < 0:
        return []
    section = full_output[idx_e : idx_r] if idx_r > idx_e else full_output[idx_e:]
    return [m.group(1) for m in INTERNAL_CITATION_PATTERN.finditer(section)]


def compute_citation_precision(
    answer: str,
    citations: List[tuple],
    retrieved_chunks: List[dict],
) -> float:
    """
    (# citations that correctly resolve to retrieved chunks) / (total citations generated).
    Uses Evidence citations only: prefers citations list (from Evidence), else parses Evidence section.
    """
    retrieved_cids = {c.get("chunk_id", "") for c in retrieved_chunks if c.get("chunk_id")}
    cids_to_check = []
    if citations:
        cids_to_check = [cid for _, cid in citations if cid]
    if not cids_to_check and "Evidence:" in answer:
        cids_to_check = parse_citations_from_evidence_section(answer)
    if not cids_to_check:
        return 1.0
    valid = sum(1 for cid in cids_to_check if cid in retrieved_cids)
    return valid / len(cids_to_check) if cids_to_check else 1.0


def compute_answer_relevance(query: str, answer: str) -> float:
    """
    Does the answer address the query intent? LLM score 1-4.
    Fallback: embedding cosine similarity mapped to 1-4 scale.
    """
    try:
        from .embed import embed_texts
        vecs = embed_texts([query, answer])
        if len(vecs) == 2 and vecs[0] and vecs[1]:
            a, b = vecs[0], vecs[1]
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na > 0 and nb > 0:
                sim = dot / (na * nb)
                return 1.0 + (sim + 1) / 2 * 3
    except Exception:
        pass
    prompt = (
        f"Query: {query}\n\nAnswer: {answer[:1500]}\n\n"
        "Does the answer address the query intent? Score 1-4: "
        "1=not relevant, 2=partially, 3=mostly, 4=fully relevant. Reply with only the number."
    )
    return _llm_score_1_4(prompt, default=2.5)


# ---- Research / Aspect metrics ----

def compute_coherence(answer: str) -> float:
    """LLM score 1-4 on logical flow and structure."""
    prompt = (
        f"Answer:\n{answer[:1500]}\n\n"
        "Rate logical flow and structure 1-4: 1=disjointed, 2=acceptable, 3=good, 4=excellent. "
        "Reply with only the number."
    )
    return _llm_score_1_4(prompt, default=2.5)


def compute_conciseness(answer: str) -> float:
    """LLM score 1-4 based on verbosity vs completeness."""
    prompt = (
        f"Answer:\n{answer[:1500]}\n\n"
        "Rate conciseness (verbosity vs completeness) 1-4: 1=too verbose or too sparse, "
        "2=acceptable, 3=good balance, 4=optimal. Reply with only the number."
    )
    return _llm_score_1_4(prompt, default=2.5)


def compute_artifact_readiness(answer: str) -> float:
    """
    Can this answer be directly converted into an evidence table or synthesis memo?
    4=Claims clearly separated, each with citation, extractable into schema.
    3=Minor restructuring needed.
    2=Claims embedded in paragraph, hard to extract.
    1=Not structurally usable.
    """
    prompt = (
        f"Answer:\n{answer[:1500]}\n\n"
        "Artifact Readiness: Can this be converted into an evidence table or synthesis memo? "
        "4=Claims clearly separated with citations, extractable. "
        "3=Minor restructuring needed. "
        "2=Claims embedded, hard to extract. "
        "1=Not usable. Reply with only the number."
    )
    return _llm_score_1_4(prompt, default=2.5)


# ---- Run evaluation for a single mode ----

def run_evaluation_single_mode(
    mode: str,
    run_fn: Callable[[str, Optional[str]], dict],
    queries_path: Path | None = None,
) -> tuple[List[dict], dict]:
    """
    Run evaluation with a given run_fn(query_text, query_id) -> result.
    Returns (per_query_metrics_list, aggregate_metrics_dict).
    """
    ensure_dirs()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_eval_queries(queries_path)
    chunk_texts = load_chunk_texts()
    per_query: List[dict] = []

    for q in queries:
        qid = q.get("query_id", "")
        qtext = q.get("query_text", "")
        try:
            result = run_fn(qtext, qid)
        except Exception as e:
            logger.warning("Query %s failed: %s", qid, e)
            per_query.append({
                "query_id": qid,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "citation_precision": 0.0,
                "answer_relevance": 0.0,
                "coherence": 0.0,
                "conciseness": 0.0,
                "artifact_readiness": 0.0,
            })
            continue

        retrieved = result.get("retrieved_chunks", [])
        citations = result.get("citations", [])
        answer = result.get("answer", "")
        retrieved_texts = [c.get("text", "") for c in retrieved]

        ctx_prec = compute_context_precision(retrieved, qtext)
        ctx_rec = compute_context_recall(retrieved, qtext)
        faith = compute_faithfulness(answer, retrieved_texts)
        cit_prec = compute_citation_precision(answer, citations, retrieved)
        ans_rel = compute_answer_relevance(qtext, answer)
        coh = compute_coherence(answer)
        conc = compute_conciseness(answer)
        art = compute_artifact_readiness(answer)

        row = {
            "query_id": qid,
            "context_precision": round(ctx_prec, 4),
            "context_recall": round(ctx_rec, 4),
            "faithfulness": round(faith, 4),
            "citation_precision": round(cit_prec, 4),
            "answer_relevance": round(ans_rel, 4),
            "coherence": round(coh, 4),
            "conciseness": round(conc, 4),
            "artifact_readiness": round(art, 4),
        }
        per_query.append(row)

        if result.get("log_record"):
            save_run_to_file(result["log_record"], metrics=row)

    n = len(per_query)
    agg = {}
    for key in [
        "context_precision", "context_recall", "faithfulness",
        "citation_precision", "answer_relevance", "coherence",
        "conciseness", "artifact_readiness",
    ]:
        vals = [r[key] for r in per_query]
        agg[f"avg_{key}"] = round(sum(vals) / n, 4) if n else 0.0

    return per_query, agg


def run_evaluation(queries_path: Path | None = None) -> tuple[dict, List[dict]]:
    """Run baseline RAG evaluation. Returns (aggregate_dict, per_query_list)."""
    def run_baseline(qtext: str, qid: str | None) -> dict:
        return run_query(qtext, query_id=qid)

    per_query, agg = run_evaluation_single_mode("baseline", run_baseline, queries_path)
    return agg, per_query


def write_eval_outputs(summary: dict, per_query: List[dict]) -> None:
    """Legacy: write to phase2_eval_summary.json and phase2_eval_per_query.csv."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    fieldnames = [
        "query_id", "context_precision", "context_recall", "faithfulness",
        "citation_precision", "answer_relevance", "coherence", "conciseness", "artifact_readiness",
    ]
    with open(EVAL_PER_QUERY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_query)
    logger.info("Wrote %s and %s", EVAL_SUMMARY_JSON, EVAL_PER_QUERY_CSV)


def write_phase2_metrics(
    per_query: List[dict],
    aggregate: dict,
    per_query_path: Path | None = None,
    aggregate_path: Path | None = None,
) -> None:
    """Write per_query_metrics.json and aggregate_metrics.json per Phase 2 spec."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    per_query_path = per_query_path or PER_QUERY_METRICS_JSON
    aggregate_path = aggregate_path or AGGREGATE_METRICS_JSON
    with open(per_query_path, "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2)
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    logger.info("Wrote %s and %s", per_query_path, aggregate_path)


def write_evaluation_summary_md(
    baseline_agg: dict,
    enhanced_agg: dict | None,
    failure_cases: List[dict],
    output_path: Path | None = None,
) -> None:
    """Write evaluation_summary.md with mean scores, 3 failure cases, comparison."""
    path = output_path or EVALUATION_SUMMARY_MD
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 Evaluation Summary",
        "",
        "## Mean Scores (Baseline RAG)",
        "",
    ]
    for k, v in sorted(baseline_agg.items()):
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Mean Scores (Enhanced RAG)", ""])
    if enhanced_agg:
        for k, v in sorted(enhanced_agg.items()):
            lines.append(f"- {k}: {v}")
        lines.extend(["", "## Comparison: Baseline vs Enhanced", ""])
        for k in baseline_agg:
            b = baseline_agg.get(k, 0)
            e = enhanced_agg.get(k, 0)
            diff = e - b
            lines.append(f"- {k}: baseline={b:.4f}, enhanced={e:.4f}, diff={diff:+.4f}")
    else:
        lines.append("(Enhanced mode not run)")
    lines.extend(["", "## Representative Failure Cases (3)", ""])
    for i, fc in enumerate(failure_cases[:3], 1):
        lines.append(f"### Failure {i}: {fc.get('query_id', '?')}")
        lines.append(f"- Query: {fc.get('query_text', '')[:200]}...")
        lines.append(f"- Issue: {fc.get('issue', 'N/A')}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)
