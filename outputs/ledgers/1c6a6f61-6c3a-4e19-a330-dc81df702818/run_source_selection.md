# Source Selection Ledger

**Run ID:** run
**Session ID:** 1c6a6f61-6c3a-4e19-a330-dc81df702818

## Selection method
- **Method:** scripted (scripted / manual / agentic)
- **Retrieval pipeline:** hybrid
- **Top-k:** 5
- **Guardrails enabled:** True

## Inclusion / Exclusion
- **Inclusion:** Chunks from indexed corpus (FAISS + BM25); bibliography/boilerplate filtered; per-source cap applied.
- **Exclusion:** Chunks below relevance threshold or bibliography score above threshold; optional topic-mismatch filter.

## Counts
- **# docs (sources) in run:** 1
- **# chunks retrieved:** 1
- **Avg chunk length (chars):** 7
- **# missing critical fields (manifest):** 0
- **Manifest total entries:** 28

## Sampling
Top-k retrieval per query; no time filter. Diversification: max N chunks per source_id, then fill by RRF score.

## Deterministic notes
Pipeline versions: Phase 2 RAG (hybrid BM25 + FAISS, RRF, two-pass for limitations). Index built from data/processed/chunks.jsonl.