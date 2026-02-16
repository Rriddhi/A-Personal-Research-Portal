# Phase 2 Prompt Kit (Revised v2 — Phase-1-Informed)

**Project:** Human-Centered, Evidence-Grounded AI for Personalized Health Recommendations  
**Course:** AI Model Development (95-864)  
**Phase:** Prompt the Research Domain (Refined)

---

## Why This Revision Exists

Phase 1 evaluation revealed consistent failure modes across models, including:
- overgeneralization beyond the population or study context described in the source
- claims stated with high confidence despite weak or ambiguous evidence
- citations that were present but weakly grounded in specific text segments
- cross-source synthesis that implicitly resolved disagreement or uncertainty

This revised prompt kit introduces targeted guardrails to address these failure modes while preserving the original task structure required by Assignment 1.

---

## Task 1: Claim–Evidence Extraction

### Task Intent
Evaluate whether models can extract only text-supported claims, explicitly represent evidence strength, and avoid speculative inference in health-related research contexts.

---

### Prompt A — Baseline (Unchanged)

```text
Extract up to five key claims from the provided text and list the evidence supporting each claim.
```

**Rationale:**  
This prompt is intentionally minimal and unchanged from Phase 1 to preserve baseline comparability.

---

### Prompt B — Structured + Evidence-Aware Guardrails (v2)

```text
You are performing a research-grade claim–evidence extraction task in a health context.

Instructions:
- Extract only claims that are explicitly supported by the provided text.
- Each claim must be directly traceable to a quoted evidence snippet.
- For each claim, assess and label the strength of evidence using one of:
  {strong, moderate, weak, unclear}.
- Cite each claim using (source_id, chunk_id).
- Do not infer, generalize, or extend findings beyond the described population,
  setting, or study design.
- If the text does not support a claim with sufficient clarity, omit it.
- If fewer than five valid claims exist, return fewer rows.

Output exactly in the following table format:

| Claim | Evidence snippet | Evidence strength | Citation (source_id, chunk_id) |
```

#### Why these constraints exist

- **Evidence strength labeling**  
  Addresses Phase 1 failures where tentative findings were presented with undue confidence.

- **Population and study-design boundaries**  
  Explicitly prevents overgeneralization beyond what the source supports.

- **Direct quotation requirement**  
  Improves traceability and citation verification.

- **Permission to omit claims**  
  Reinforces that restraint is preferable to hallucination.

---

## Task 2: Cross-Source Synthesis

### Task Intent
Evaluate whether models can synthesize across sources while preserving disagreement, uncertainty, and asymmetries in evidence quality.

---

### Prompt A — Baseline (Unchanged)

```text
Compare the two provided sources and summarize where they agree and disagree.
```

**Rationale:**  
Retained unchanged to allow comparison with Phase 1 baseline synthesis behavior.

---

### Prompt B — Structured + Disagreement-Preserving Guardrails (v2)

```text
You are synthesizing evidence across multiple research sources in a health domain.

Instructions:
- Identify explicit points of agreement and disagreement between the sources.
- Support each point with quoted evidence from each source.
- If one source provides stronger, weaker, or more limited evidence than the other,
  state this explicitly.
- If findings conflict or evidence is inconclusive, say so clearly.
- Do not resolve disagreements or recommend a single conclusion unless explicitly
  supported by the sources.
- Do not introduce external knowledge or assumptions.

Output exactly in the following table format:

| Aspect | Source A position (with citation) | Source B position (with citation) | Notes on uncertainty or evidence strength |
```

#### Why these constraints exist

- **Asymmetric evidence handling**  
  Prevents false equivalence between sources with different levels of rigor.

- **Explicit uncertainty column**  
  Forces acknowledgment of ambiguity rather than smoothing over conflict.

- **No-resolution rule**  
  Directly counters the tendency toward forced consensus observed in Phase 1.

---

## Updated Prompt Design Philosophy

This revised prompt kit reflects three key lessons from Phase 1:

1. **Grounding must be explicit, not implicit**  
   Claims must be directly supported by quoted evidence and precise citations.

2. **Uncertainty is first-class information**  
   Weak, missing, or conflicting evidence is a valid and necessary output.

3. **Restraint is a success condition**  
   Producing fewer claims or stating “not supported” is preferable to confident fabrication.

---

## Relationship to Phase 2

- Evidence-strength labels map cleanly to confidence or evidence-scoring mechanisms.
- Population-bounded claims support safer retrieval-augmented generation.
- Disagreement-preserving synthesis informs multi-document RAG responses.
- Citation discipline aligns with manifest-backed chunk referencing.

---

## Summary

This revised prompt kit preserves the structure required by Assignment 1 while incorporating empirically motivated guardrails derived from Phase 1 evaluation. It provides a stronger foundation for human-centered, evidence-grounded AI systems and is directly reusable in Phase 2 generation pipelines.
