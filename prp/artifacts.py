"""
Phase 3: Artifact generation — evidence table (required) and synthesis memo (optional).
Evidence table schema: Claim | Evidence snippet | Citation | Confidence | Notes [+ support_type, source_diversity].
"""
import csv
import io
import re
from pathlib import Path
from typing import Callable, List, Optional

from .config import OUTPUTS_ARTIFACTS_DIR, ensure_dirs
from .citations import format_in_text_apa, get_source_meta
from .generate import normalize_chunk_citation_for_display
from .utils import clean_chunk_text_for_generation, fix_concatenated_words, normalize_claim_spacing

# Sentence split for claim extraction
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize_claim(claim: str) -> str:
    """Apply spacing fix and concatenation fixes so claims display cleanly."""
    if not claim or not claim.strip():
        return claim
    s = normalize_claim_spacing(claim.strip())
    return fix_concatenated_words(s)


def _extract_answer_body(full_output: str) -> str:
    """Extract only the Answer section prose (no dashes, no Evidence/References)."""
    if not full_output or not isinstance(full_output, str):
        return ""
    s = full_output.strip()
    # Find "Answer:" and take until "Evidence:" or "References:"
    idx_a = s.find("Answer:")
    if idx_a >= 0:
        s = s[idx_a + 7 :].lstrip()
    for marker in ("Evidence:", "References:", "Reference-to-Chunk"):
        idx = s.find(marker)
        if idx > 0:
            s = s[:idx]
    # Strip leading dashes/equals and blank lines
    s = re.sub(r"^[-=\s]+", "", s)
    return s.strip()


def _extract_evidence_section(full_output: str) -> str:
    """Extract the Evidence section (numbered quotes) from full answer text, or empty if not present."""
    if not full_output or not isinstance(full_output, str):
        return ""
    s = full_output.strip()
    idx_e = s.find("Evidence:")
    if idx_e < 0:
        return ""
    s = s[idx_e + 9 :].lstrip()
    for marker in ("References:", "Reference-to-Chunk", "Limitations"):
        idx = s.find(marker)
        if idx > 0:
            s = s[:idx]
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _sentences_from_answer(text: str) -> List[str]:
    """Split answer into sentences (min length 40 chars)."""
    if not text or not isinstance(text, str):
        return []
    raw = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip() and len(s.strip()) >= 40]


# Meta / vague sentences: about what passages don't say, or "they discuss" without a clear claim. We skip these.
_META_CLAIM_STARTS = (
    "the corpus does not",
    "the corpus does not contain",
    "the retrieved sources",
    "the material is useful",
    "what the corpus does not",
    "direct evidence for the exact question is limited",
    "these passages provide",
    "the evidence is drawn",
    "evidence that would be needed",
    "primary research addressing",
    "systematic reviews or meta-analyses",
    "the passages do not",
    "the passages do not specifically",
    "however, they discuss",
    "additionally, they mention",
    "overall, while the passages",
)
# Minimum fraction of claim words that must appear in the snippet for the row to be included (avoids mismatched pairs)
_MIN_CLAIM_SNIPPET_OVERLAP = 0.22


def _is_meta_claim(sentence: str) -> bool:
    """True if the sentence is about the corpus/retrieval or is a vague 'they discuss' rather than a clear claim."""
    s = (sentence or "").strip().lower()
    if len(s) < 30:
        return True
    for start in _META_CLAIM_STARTS:
        if s.startswith(start):
            return True
    return False


def _claim_snippet_overlap(claim: str, snippet: str) -> float:
    """Fraction of claim's meaningful words (len>3) that appear in snippet. Used to avoid showing unsupported claim–snippet pairs."""
    if not claim or not snippet:
        return 0.0
    words = set(w for w in re.findall(r"\w+", (claim or "").lower()) if len(w) > 3)
    if not words:
        return 0.0
    snip_lower = (snippet or "").lower()
    hit = sum(1 for w in words if w in snip_lower)
    return hit / len(words)


def _claim_from_snippet(snippet: str, max_len: int = 5000) -> str:
    """
    Derive a claim-like sentence from an evidence snippet (full first sentence, no truncation).
    Avoids mid-sentence fragments (PDF artifacts) and list items; prefers a complete declarative sentence.
    """
    if not snippet or not snippet.strip():
        return ""
    s = clean_chunk_text_for_generation(snippet).strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) < 25:
        return s[:max_len]
    # Skip leading fragment: find first uppercase (sentence start) or first ". " and take what follows
    start = 0
    for i, c in enumerate(s):
        if c.isupper() and (i == 0 or (i >= 1 and s[i - 1] in " \t.")):
            start = i
            break
    if start > 0:
        s = s[start:]
    # If still starts with list-item or mid-phrase, try to take after first ". "
    if s.lower().startswith(("(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)", "and (", "e ", "th ")):
        idx = s.find(". ", 3)
        if idx > 5:
            s = s[idx + 2 :].lstrip()
    # First sentence only — return it in full so claims display completely
    first_sentence = s
    for sep in (". ", "。", "! ", "? "):
        idx = s.find(sep)
        if idx >= 10:
            first_sentence = s[: idx + 1].strip()
            break
    s = first_sentence.strip() if first_sentence.strip() else s[:max_len]
    # Only truncate if absurdly long (e.g. no sentence boundary found)
    if len(s) > max_len:
        s = s[: max_len - 3].rsplit(" ", 1)[0] + "..."
    return s


def _best_snippet_for_claim(claim: str, chunks: List[dict], citation_formatter: Optional[Callable] = None) -> tuple:
    """
    Choose best supporting chunk by keyword overlap (simple).
    Returns (evidence_snippet, citation, score, best_chunk).
    citation_formatter(source_id, chunk_id) -> str; default uses normalize_chunk_citation_for_display + APA.
    """
    if not chunks:
        return ("", "", 0.0, None)
    claim_lower = (claim or "").lower()
    words = set(w for w in re.findall(r"\w+", claim_lower) if len(w) > 3)
    if not words:
        c = chunks[0]
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        raw = (c.get("text") or "")[:400]
        text = clean_chunk_text_for_generation(raw)
        cit = (citation_formatter or _default_citation_fmt)(sid, cid)
        return (text, cit, 0.5, c)
    best_score = 0.0
    best_snippet = ""
    best_cit = ""
    best_chunk = None
    for c in chunks:
        text = (c.get("text") or "").lower()
        overlap = sum(1 for w in words if w in text)
        score = overlap / max(len(words), 1)
        rrf = c.get("rrf_score") or c.get("vector_score") or 0.5
        combined = 0.6 * score + 0.4 * min(1.0, rrf) if isinstance(rrf, (int, float)) else score
        if combined > best_score:
            best_score = combined
            raw = (c.get("text") or "")[:400]
            snippet = clean_chunk_text_for_generation(raw)
            # Prefer a window around first matched term
            full_text = c.get("text") or ""
            for w in words:
                if w in snippet.lower():
                    idx = snippet.lower().find(w)
                    start = max(0, idx - 80)
                    end = min(len(snippet), idx + 200)
                    snippet = (snippet[start:end] + ("..." if end < len(full_text) else "")).strip()
                    break
            best_snippet = snippet or clean_chunk_text_for_generation((full_text or "")[:300])
            sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
            best_cit = (citation_formatter or _default_citation_fmt)(sid, cid)
            best_chunk = c
    if best_chunk is None and chunks:
        c = chunks[0]
        best_snippet = clean_chunk_text_for_generation((c.get("text") or "")[:300])
        best_cit = (citation_formatter or _default_citation_fmt)(c.get("source_id", ""), c.get("chunk_id", ""))
        best_score = 0.3
        best_chunk = c
    return (best_snippet, best_cit, best_score, best_chunk)


def _default_citation_fmt(source_id: str, chunk_id: str) -> str:
    """Citation: paper title + APA, or APA only if no title."""
    try:
        meta = get_source_meta(source_id or "")
        title = (meta.get("title") or "").strip()
        apa = format_in_text_apa(source_id or "", meta)
        if title:
            short_title = title[:60] + "..." if len(title) > 60 else title
            return f"{short_title} · {apa}"
        return apa
    except Exception:
        return format_in_text_apa(source_id or "")


def _confidence_label(score: float) -> str:
    if score >= 0.7:
        return "H"
    if score >= 0.4:
        return "M"
    return "L"


def build_evidence_table(
    answer_text: str,
    citations: List[tuple],
    retrieved_chunks: List[dict],
    citation_formatter: Optional[Callable[[str, str], str]] = None,
    max_claims: int = 20,
) -> List[dict]:
    """
    Build evidence table rows: claim, evidence snippet, citation, confidence, notes.
    Extracts 5–10 informative sentences as claims; matches to best supporting chunk via keyword overlap.
    Uses only the Answer section body (not Evidence/References) for claim extraction.
    """
    answer_body = _extract_answer_body(answer_text) or answer_text or ""
    rows = []
    sentences = _sentences_from_answer(answer_body)
    skip_headers = ("answer:", "evidence:", "references:", "reference-to-chunk", "----------")
    # Prefer substantive claims: drop meta sentences that only describe the corpus/retrieval
    candidate_claims = [
        s for s in (sentences[: max_claims * 3] if sentences else [])
        if len(s.strip()) > 25
        and not s.strip().lower().startswith(skip_headers)
        and not _is_meta_claim(s)
    ][:max_claims]
    cited_ids = {c[1] for c in citations if len(c) >= 2}
    chunk_by_id = {c.get("chunk_id", ""): c for c in retrieved_chunks if c.get("chunk_id")}
    cited_chunks = [chunk_by_id[cid] for cid in cited_ids if cid in chunk_by_id]
    if not cited_chunks:
        cited_chunks = retrieved_chunks[:10]

    min_rows = min(10, max_claims)  # aim for at least 10 rows when we have enough chunks
    used_chunk_ids = set()

    if candidate_claims:
        # Build rows from answer-derived claims only when the snippet actually supports the claim (min word overlap)
        for claim in candidate_claims:
            claim_clean = _normalize_claim(claim)
            snippet, citation, score, best_chunk = _best_snippet_for_claim(claim_clean, cited_chunks or retrieved_chunks, citation_formatter)
            overlap = _claim_snippet_overlap(claim_clean, snippet)
            if overlap < _MIN_CLAIM_SNIPPET_OVERLAP:
                continue  # skip: this snippet doesn't support this claim
            if best_chunk and best_chunk.get("chunk_id"):
                used_chunk_ids.add(best_chunk["chunk_id"])
            confidence = _confidence_label(score)
            support = "direct quote" if score >= 0.6 else ("paraphrase" if score >= 0.35 else "inferred")
            notes = support + (" (weak support)" if score < 0.35 else "")
            rows.append({
                "Claim": claim_clean,
                "Evidence snippet": snippet,
                "Citation": citation,
                "Confidence": confidence,
                "Notes": notes,
                "support_type": support,
                "source_diversity": "",
            })
        # If we have too few rows (e.g. only 1 claim), supplement with rows derived from other cited chunks
        chunks_pool = cited_chunks or retrieved_chunks
        for c in (chunks_pool or [])[:max_claims * 2]:
            if len(rows) >= max_claims:
                break
            cid = c.get("chunk_id")
            if not cid or cid in used_chunk_ids:
                continue
            raw_text = (c.get("text") or "").strip()
            if len(raw_text) < 30:
                continue
            text = clean_chunk_text_for_generation(raw_text)
            claim = _claim_from_snippet(text)
            if not claim:
                continue
            used_chunk_ids.add(cid)
            sid = c.get("source_id", "")
            snippet = text[:400] if len(text) > 400 else text
            citation = (citation_formatter or _default_citation_fmt)(sid, cid)
            score = 0.5 if c.get("rrf_score") or c.get("vector_score") else 0.4
            rows.append({
                "Claim": _normalize_claim(claim),
                "Evidence snippet": snippet,
                "Citation": citation,
                "Confidence": _confidence_label(score),
                "Notes": "Claim derived from source; supports understanding of the evidence.",
                "support_type": "direct quote",
                "source_diversity": "",
            })
    else:
        # No substantive claims in the answer: derive claims from the evidence itself (what the sources say)
        chunks = cited_chunks or retrieved_chunks[:max_claims]
        for c in chunks[:max_claims]:
            raw_text = (c.get("text") or "").strip()
            if not raw_text or len(raw_text) < 30:
                continue
            text = clean_chunk_text_for_generation(raw_text)
            claim = _claim_from_snippet(text)
            if not claim:
                continue
            sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
            snippet = text[:400] if len(text) > 400 else text
            citation = (citation_formatter or _default_citation_fmt)(sid, cid)
            score = 0.5 if c.get("rrf_score") or c.get("vector_score") else 0.4
            confidence = _confidence_label(score)
            notes = "Claim derived from source; supports understanding of the evidence."
            rows.append({
                "Claim": _normalize_claim(claim),
                "Evidence snippet": snippet,
                "Citation": citation,
                "Confidence": confidence,
                "Notes": notes,
                "support_type": "direct quote",
                "source_diversity": "",
            })
    if not rows and answer_body:
        # Fallback: one row so the table is not empty (e.g. answer was all meta)
        first_chunk = (cited_chunks or retrieved_chunks)[0] if (cited_chunks or retrieved_chunks) else {}
        sid, cid = first_chunk.get("source_id", ""), first_chunk.get("chunk_id", "")
        claim = _normalize_claim(answer_body[:500].strip()) + ("..." if len(answer_body) > 500 else "")
        raw_snippet = (first_chunk.get("text") or "")[:400] if first_chunk else answer_body[:400]
        snippet = clean_chunk_text_for_generation(raw_snippet) if first_chunk else fix_concatenated_words(raw_snippet)
        citation = (citation_formatter or _default_citation_fmt)(sid, cid) if sid else "—"
        rows.append({
            "Claim": claim,
            "Evidence snippet": snippet,
            "Citation": citation,
            "Confidence": "medium",
            "Notes": "Summary (no substantive claims extracted)",
            "support_type": "inferred",
            "source_diversity": "",
        })
    return rows


def _cell_md(text: str, max_len: int = 400) -> str:
    """Escape pipes and newlines for markdown table cell."""
    if not text:
        return ""
    s = (text[:max_len] + ("..." if len(text) > max_len else "")).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", s).strip()


def _escape_html(text: str, max_len: int = 600) -> str:
    """Escape for HTML content; truncate to max_len."""
    if not text:
        return ""
    s = (text[:max_len] + ("..." if len(text) > max_len else "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return s.replace("\n", " ").replace("\r", " ").strip()


def render_evidence_table_html(rows: List[dict], claim_len: int = 5000, snippet_len: int = 2000, citation_len: int = 180) -> str:
    """Render evidence table as HTML so it displays as a real table (e.g. in Streamlit with unsafe_allow_html)."""
    if not rows:
        return "<table><tr><th>#</th><th>Claim</th><th>Evidence snippet</th><th>Citation</th><th>Confidence</th><th>Notes</th></tr></table>"
    # Column width hints so Claim and Evidence snippet get enough space (table-layout: fixed in CSS)
    lines = [
        "<table>",
        "<colgroup><col style=\"width:3%\"><col style=\"width:24%\"><col style=\"width:35%\"><col style=\"width:20%\"><col style=\"width:6%\"><col style=\"width:12%\"></colgroup>",
        "<tr><th>#</th><th>Claim</th><th>Evidence snippet</th><th>Citation</th><th>Confidence</th><th>Notes</th></tr>",
    ]
    for i, r in enumerate(rows, 1):
        claim = _escape_html(r.get("Claim") or "", claim_len)
        snippet = _escape_html(r.get("Evidence snippet") or "", snippet_len)
        cit = _escape_html(r.get("Citation") or "", citation_len)
        conf = _escape_html((r.get("Confidence") or "").strip(), 20)
        notes = _escape_html(r.get("Notes") or "", 200)
        lines.append(f"<tr><td>{i}</td><td>{claim}</td><td>{snippet}</td><td>{cit}</td><td>{conf}</td><td>{notes}</td></tr>")
    lines.append("</table>")
    return "\n".join(lines)


def render_evidence_table_md(rows: List[dict], claim_len: int = 5000, snippet_len: int = 2000, citation_len: int = 180) -> str:
    """Render evidence table as Markdown (one row per data row; no newlines in cells)."""
    if not rows:
        return "| # | Claim | Evidence snippet | Citation | Confidence | Notes |\n|---|-------|------------------|----------|------------|-------|\n"
    lines = ["| # | Claim | Evidence snippet | Citation | Confidence | Notes |", "|---|-------|------------------|----------|------------|-------|"]
    for i, r in enumerate(rows, 1):
        claim = _cell_md(r.get("Claim") or "", claim_len)
        snippet = _cell_md(r.get("Evidence snippet") or "", snippet_len)
        cit = _cell_md(r.get("Citation") or "", citation_len)
        conf = (r.get("Confidence") or "").strip()
        notes = _cell_md(r.get("Notes") or "", 200)
        lines.append(f"| {i} | {claim} | {snippet} | {cit} | {conf} | {notes} |")
    return "\n".join(lines)


def render_evidence_table_csv(rows: List[dict]) -> str:
    """Render evidence table as CSV string."""
    if not rows:
        return "#,Claim,Evidence snippet,Citation,Confidence,Notes,support_type,source_diversity\n"
    fieldnames = ["#", "Claim", "Evidence snippet", "Citation", "Confidence", "Notes", "support_type", "source_diversity"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for i, r in enumerate(rows, 1):
        row = {"#": i, "Claim": r.get("Claim") or "", "Evidence snippet": r.get("Evidence snippet") or "",
               "Citation": r.get("Citation") or "", "Confidence": r.get("Confidence") or "",
               "Notes": r.get("Notes") or "", "support_type": r.get("support_type") or "",
               "source_diversity": r.get("source_diversity") or ""}
        w.writerow(row)
    return buf.getvalue()


def _format_memo_body(text: str, max_len: int = 2500) -> str:
    """Add paragraph breaks so memo is not one chunk. List items get newlines; sentences get space."""
    if not text or not text.strip():
        return ""
    s = text[:max_len].strip()
    # Put list items on their own lines: " - " or " -" after a period/newline
    s = re.sub(r"\.\s+(-\s+)", r".\n\n\1", s)
    s = re.sub(r"(\w)\s+(-\s+[A-Z])", r"\1\n\n\2", s)
    # Break at common section-like phrases so they start a new paragraph
    for phrase in ("What the corpus does NOT", "Evidence that would be", "Direct evidence", "The retrieved sources"):
        if phrase in s:
            s = s.replace(phrase, "\n\n" + phrase)
    # Collapse triple+ newlines to double
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_synthesis_memo(
    answer_text: str,
    citations: List[tuple],
    retrieved_chunks: List[dict],
    citation_formatter: Optional[Callable[[str, str], str]] = None,
    word_target: tuple = (800, 1200),
) -> str:
    """
    Build synthesis memo with Summary, Evidence, Limitations, References.
    Summary = short cap of answer prose; Evidence = extracted Evidence section or bullet points from chunks (not the same as Summary).
    """
    body = _extract_answer_body(answer_text) or (answer_text or "")
    # Summary: brief (first ~120 words of answer prose only)
    words = body.split()
    summary = " ".join(words[: min(len(words), 120)])
    if len(words) > 120:
        summary = summary.rstrip() + " …"
    summary_formatted = _format_memo_body(summary, max_len=800)
    # Evidence: use the Evidence section from the full answer if present; else build from chunks
    evidence_block = _extract_evidence_section(answer_text)
    if evidence_block:
        evidence_formatted = _format_memo_body(evidence_block, max_len=2500)
    else:
        # Build from retrieved chunks: short bullet points with snippet + citation
        fmt = citation_formatter or _default_citation_fmt
        bullets = []
        seen = set()
        for c in (retrieved_chunks or [])[:8]:
            cid = c.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            text = clean_chunk_text_for_generation((c.get("text") or "").strip())
            if len(text) < 30:
                continue
            first_sent = text.split(". ")[0].strip() + ("." if not text.split(". ")[0].strip().endswith(".") else "")
            first_sent = first_sent[:200] + ("…" if len(first_sent) > 200 else "")
            sid = c.get("source_id", "")
            apa = format_in_text_apa(sid, get_source_meta(sid)) if sid else ""
            bullets.append(f"- {first_sent} ({apa})")
        evidence_formatted = "\n\n".join(bullets) if bullets else "No evidence excerpts available."
    ref_ids = list(dict.fromkeys(sid for sid, _ in citations if sid))
    ref_section = []
    for sid in ref_ids:
        try:
            meta = get_source_meta(sid)
            title = (meta.get("title") or "").strip() or sid
            apa = format_in_text_apa(sid, meta)
            ref_section.append(f"- {apa}. {title}")
        except Exception:
            ref_section.append(f"- {format_in_text_apa(sid)}. {sid}")
    ref_block = "\n\n".join(ref_section) if ref_section else "No references."
    memo = f"""## Summary

{summary_formatted or summary}


## Evidence

{evidence_formatted}


## Limitations / Uncertainties

The evidence is drawn from the retrieved corpus; gaps and conflicting findings are noted in the answer. Interpret with care.


## References

{ref_block}
"""
    return memo


def save_evidence_table_artifact(
    session_id: str,
    run_id: str,
    rows: List[dict],
    formats: List[str],
) -> List[dict]:
    """Save evidence table under outputs/artifacts/{session_id}/{run_id}/. Returns list of {type, path, format}."""
    ensure_dirs()
    base = OUTPUTS_ARTIFACTS_DIR / session_id / run_id
    base.mkdir(parents=True, exist_ok=True)
    artifacts = []
    if "md" in formats or "markdown" in formats:
        path = base / "evidence_table.md"
        path.write_text(render_evidence_table_md(rows), encoding="utf-8")
        artifacts.append({"type": "evidence_table", "path": str(path), "format": "md"})
    if "csv" in formats:
        path = base / "evidence_table.csv"
        path.write_text(render_evidence_table_csv(rows), encoding="utf-8")
        artifacts.append({"type": "evidence_table", "path": str(path), "format": "csv"})
    return artifacts


def save_synthesis_memo_artifact(
    session_id: str,
    run_id: str,
    memo_md: str,
) -> dict:
    """Save synthesis memo under outputs/artifacts/{session_id}/{run_id}/synthesis_memo.md."""
    ensure_dirs()
    base = OUTPUTS_ARTIFACTS_DIR / session_id / run_id
    base.mkdir(parents=True, exist_ok=True)
    path = base / "synthesis_memo.md"
    path.write_text(memo_md, encoding="utf-8")
    return {"type": "synthesis_memo", "path": str(path), "format": "md"}
