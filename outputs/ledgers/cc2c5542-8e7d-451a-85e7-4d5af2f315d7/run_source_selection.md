# Source Selection Ledger

**Run ID:** run
**Session ID:** cc2c5542-8e7d-451a-85e7-4d5af2f315d7

## Selection method
- **Method:** scripted (scripted / manual / agentic)
- **Retrieval pipeline:** hybrid_bm25_faiss_rrf
- **Top-k:** 5
- **Guardrails enabled:** True

## Inclusion / Exclusion
- **Inclusion:** Chunks from indexed corpus (FAISS + BM25); bibliography/boilerplate filtered; per-source cap applied.
- **Exclusion:** Chunks below relevance threshold or bibliography score above threshold; optional topic-mismatch filter.

## Counts
- **# docs (sources) in run:** 1
- **# chunks retrieved:** 1
- **Avg chunk length (chars):** 1
- **# missing critical fields (manifest):** 0
- **Manifest total entries:** 28

## Sampling
Top-k retrieval per query; no time filter. Diversification: max N chunks per source_id, then fill by RRF score.

## Deterministic notes
Pipeline versions: Phase 2 RAG (hybrid BM25 + FAISS, RRF, two-pass for limitations). Index built from data/processed/chunks.jsonl.