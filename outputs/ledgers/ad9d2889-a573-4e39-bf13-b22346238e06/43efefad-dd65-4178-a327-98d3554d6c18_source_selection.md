# Source Selection Ledger

**Run ID:** 43efefad-dd65-4178-a327-98d3554d6c18
**Session ID:** ad9d2889-a573-4e39-bf13-b22346238e06

## Selection method
- **Method:** scripted (scripted / manual / agentic)
- **Retrieval pipeline:** RRF (hybrid)
- **Top-k:** 15
- **Guardrails enabled:** True

## Inclusion / Exclusion
- **Inclusion:** Chunks from indexed corpus (FAISS + BM25); bibliography/boilerplate filtered; per-source cap applied.
- **Exclusion:** Chunks below relevance threshold or bibliography score above threshold; optional topic-mismatch filter.

## Counts
- **# docs (sources) in run:** 9
- **# chunks retrieved:** 15
- **Avg chunk length (chars):** 0
- **# missing critical fields (manifest):** 0
- **Manifest total entries:** 28

## Sampling
Top-k retrieval per query; no time filter. Diversification: max N chunks per source_id, then fill by RRF score.

## Deterministic notes
Pipeline versions: Phase 2 RAG (hybrid BM25 + FAISS, RRF, two-pass for limitations). Index built from data/processed/chunks.jsonl.