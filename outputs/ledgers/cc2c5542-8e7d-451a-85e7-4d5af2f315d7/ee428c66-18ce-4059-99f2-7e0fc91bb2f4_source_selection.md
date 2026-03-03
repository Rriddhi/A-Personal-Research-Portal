# Source Selection Ledger

**Run ID:** ee428c66-18ce-4059-99f2-7e0fc91bb2f4
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
- **# docs (sources) in run:** 8
- **# chunks retrieved:** 10
- **Avg chunk length (chars):** 0
- **# missing critical fields (manifest):** 0
- **Manifest total entries:** 28

## Sampling
Top-k retrieval per query; no time filter. Diversification: max N chunks per source_id, then fill by RRF score.

## Deterministic notes
Pipeline versions: Phase 2 RAG (hybrid BM25 + FAISS, RRF, two-pass for limitations). Index built from data/processed/chunks.jsonl.