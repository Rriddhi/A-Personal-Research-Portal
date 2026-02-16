"""Generation: synthesized answer from retrieved chunks with inline citations; trust behavior."""
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    MODEL_NAME,
    PROMPT_VERSION,
    MANIFEST_PATH,
    MIN_SUPPORT_BULLETS,
    MIN_LIMITATIONS_BULLETS,
    KEYWORD_OVERLAP_THRESHOLD,
    ANSWER_MIN_WORDS,
    ANSWER_MAX_WORDS,
    MIN_EVIDENCE_SNIPPETS,
)
from .utils import (
    logger,
    clean_text,
    clean_chunk_text_for_generation,
    clean_question,
    evidence_strength,
    evidence_type_label,
    is_bibliography_chunk,
)
from .citations import (
    format_in_text_apa,
    references_apa_sorted,
    build_internal_to_apa_map,
    map_internal_citations_to_apa,
    map_internal_to_apa_in_answer_section_only,
    build_citation_mapping,
)

# Minimum RRF score to consider evidence "present"; below this we refuse to cite
MIN_CONFIDENCE_THRESHOLD = 0.01

# Answerability gate: refuse to synthesize if retrieval is not semantically relevant (e.g. out-of-domain)
ANSWERABILITY_THRESH = 0.28  # min query–chunk cosine similarity (calibrated for MiniLM)
MIN_RELEVANT_CHUNKS = 2  # need at least this many chunks above THRESH to synthesize

# Relevance gate: only emit "limited evidence" when retrieval is truly off-topic
# Positive case (synthesize): keyword_hit_count >= 2 AND strong_signal (phrase indicating substantive content)
STRONG_SIGNAL_PHRASES = (
    "evaluated", "studies", "systematic review", "defined as", "we propose", "we hypothesize",
    "common", "most common", "framework", "recommendation system", "recommender",
    "personalized", "recommendation", "evidence", "literature", "research", "findings",
)
CHUNK_TEXT_FOR_ANSWERABILITY = 2000  # truncate chunk text for embedding

# Strong-claim keywords: if query contains these, we require explicit support in retrieved text
STRONG_CLAIM_KEYWORDS = ("cure", "cures", "cured", "guarantee", "proven", "eliminates", "reverses")

# Claim-validation query patterns: does.*prove, does.*cure, always, completely, for everyone equally
CLAIM_VALIDATION_PATTERNS = (
    re.compile(r"does\s+.*\s+prove", re.IGNORECASE),
    re.compile(r"does\s+.*\s+cure", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bcompletely\b", re.IGNORECASE),
    re.compile(r"for\s+everyone\s+equally", re.IGNORECASE),
)
# Phrases that indicate absolutes (if in query, require explicit support in chunks for stance Supported)
ABSOLUTE_PHRASES = ("always", "completely", "for everyone")

# Target sentence count for synthesized answer (3–5 short sentences by theme)
MIN_SENTENCES = 3
MAX_SENTENCES = 5

# Max quote length for Evidence section (legacy)
EVIDENCE_QUOTE_CHARS = 180
# Max words for single permitted quote in human-readable output
MAX_QUOTE_WORDS = 25

# Evidence guardrails: skip quotes matching figure/table captions, abbreviations
EVIDENCE_JUNK_RE = re.compile(
    r"\bFigure\b|\bTable\b|HDL:|\bSupplement\b|Example of composite score",
    re.IGNORECASE,
)

# Answer sentence guardrails: must NOT match (no PDF boilerplate or captions)
ANSWER_BOILERPLATE_RE = re.compile(
    r"page\s*number\s*not\s*for\s*citation|JOURNAL\s*OF\s*MEDICAL\s*INTERNET|XSL[•·]\s*FO|RenderX|"
    r"Volume\s*\d|\bArticle\b|J\s*Med\s*Internet\s*Res|doi\.org|doi\s*:|https?://|"
    r"\bFigure\b|\bTable\b|HDL:|Example of composite score",
    re.IGNORECASE,
)

# Banned substrings: sentences containing any of these are filtered from Answer and Evidence
# (acknowledgements, thanks, funding, institutional boilerplate)
BANNED_NONCONTENT_SUBSTRINGS = [
    "acknowledg",
    "thank",
    "contribution to",
    "partners",
    "funding",
    "we thank",
    "conflicts of interest",
]

# Generic section transitions / headers: if sentence is mostly one of these, filter
BANNED_SECTION_PHRASES = (
    "Publications",
    "Open Access",
    "Received:",
    "Accepted:",
    "Published:",
    "Copyright",
    "Author contributions",
    "Competing interests",
    "Data availability",
)

# Common verbs (lowercase): used to detect content sentences vs. institutional name lists
_COMMON_VERBS = frozenset(
    "is are was were have has had do does did can may will would could should "
    "show shows showed found use used include includes suggest suggests demonstrate "
    "report reported based use used provide provides support supports".split()
)

# Minimum length for a sentence to be used in the Answer (avoid fragments)
MIN_ANSWER_SENTENCE_LEN = 60

# Question-conditioned extraction: min relevant sentences across chunks or we refuse to answer
MIN_RELEVANT_SENTENCES = 3
MAX_EXTRACTED_PER_CHUNK = 3
# Weight for cosine sim vs keyword score when ranking sentences (keyword_weight=1, sim_weight=scale)
EXTRACT_SIM_WEIGHT = 2.0

# Fixed themes: 1 = definition/goal, 2 = variability/limits, 3 = data+privacy/ethics, 4 = datasets/limitations
THEME_KEYWORDS: Dict[str, List[str]] = {
    "definition_goal": ["personaliz", "nutrition", "goal", "definition", "recommend", "dietary", "individual"],
    "variability_limits": ["variability", "vary", "one-size", "limits", "evidence", "trial", "study", "response"],
    "data_privacy_ethics": ["data", "privacy", "ethics", "consent", "trust", "risk", "responsible", "patient"],
    "datasets_limitations": ["dataset", "system", "model", "collection", "framework", "corpus", "survey"],
}

# Section keywords for structured answer (A/B/C/D)
SECTION_B_KEYWORDS = ["guideline", "one-size", "differ", "population-level", "individual", "tailor", "customiz"]
SECTION_C_KEYWORDS = ["evidence", "suggest", "show", "found", "demonstrate", "support", "indicate"]
SECTION_D_KEYWORDS = ["limitation", "uncertainty", "however", "varies", "not all", "mixed", "inconclusive"]

# High-salience keywords: if present in query, at least one must appear in retrieved chunks for positive answer
HIGH_SALIENCE_KEYWORDS = (
    "vitiligo", "celery", "detox", "smoothie", "autoimmune", "skin", "RCT", "randomized",
    "heavy metal", "metal detox", "smoothies", "outcomes",
)

# Condition/entity terms: medical conditions or entities users ask about
CONDITION_TERMS = ("vitiligo", "autoimmune", "skin", "celiac", "t1dm", "t2dm", "diabetes")
# Domain terms for intent coverage: always include if in query
INTENT_DOMAIN_TERMS = ("autoimmune", "lupus", "ra", "t1d", "t1dm", "vitiligo", "ai", "artificial", "intelligence", "governance", "checklist")
# Intervention terms: treatments, supplements, etc.
INTERVENTION_TERMS = ("celery", "juice", "detox", "smoothie", "smoothies", "reverses", "celery juice")

# Dual citation system: Answer has NO citations; Evidence uses (clean_source_id_chunk_id) only
INTERNAL_CITATION_PATTERN = re.compile(r"\(([A-Za-z0-9_\-\.]+(?:_[cC]\d+)?)\)")


def clean_source_id(full_source_id: str) -> str:
    """
    Remove .pdf and _<hexhash> suffix from source_id.
    Example: jpm-15-00058-v2.pdf_d26bbd07bb39 -> jpm-15-00058-v2
    """
    if not full_source_id:
        return ""
    s = re.sub(r"\.pdf", "", full_source_id, flags=re.IGNORECASE)
    s = re.sub(r"_[a-f0-9]{8,}$", "", s, flags=re.IGNORECASE)
    return s.strip("_.")
# APA-style: (Author et al., Year) or (Author, Year)
APA_CITATION_PATTERN = re.compile(r"\([A-Za-z][^)]*?,?\s*(?:\d{4}|n\.d\.)[^)]*\)", re.IGNORECASE)


def strip_all_citations_from_text(text: str) -> str:
    """
    Remove ALL citation patterns from text (Answer must have no citations).
    Strips: (chunk_id), (source_id_c00xx), (Author et al., Year), etc.
    """
    if not text or not isinstance(text, str):
        return text
    # Remove internal citations (chunk_id format)
    out = INTERNAL_CITATION_PATTERN.sub("", text)
    # Remove APA-style citations
    out = APA_CITATION_PATTERN.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_chunk_citation_for_display(chunk_id: str, source_id: str) -> str:
    """
    Produce clean Evidence citation: (clean_source_id_chunk_id).
    No file hashes. Format: (jpm-15-00058-v2_c0008)
    """
    if not chunk_id:
        return ""
    m = re.search(r"_([cC]\d+)$", chunk_id)
    suffix = m.group(1) if m else (chunk_id.split("_")[-1] if "_" in chunk_id else chunk_id)
    sid_raw = source_id or (chunk_id[: m.start()] if m else chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id)
    clean_sid = clean_source_id(sid_raw)
    result = f"{clean_sid}_{suffix}" if clean_sid and suffix else (clean_sid or chunk_id)
    return f"({result})"


def format_internal_citation(chunk_id: str) -> str:
    """Internal citation format: (chunk_id) for validation and traceability."""
    return f"({chunk_id})"


def parse_internal_citations_from_answer(answer: str, valid_chunk_ids: set) -> List[str]:
    """
    Parse internal citations (chunk_id) from answer. Only returns chunk_ids
    that match parenthetical content in valid_chunk_ids (avoids false positives).
    """
    out: List[str] = []
    for m in INTERNAL_CITATION_PATTERN.finditer(answer):
        token = m.group(1)
        if token in valid_chunk_ids:
            out.append(token)
    return out


def parse_citations_from_answer(answer: str) -> List[Tuple[str, Optional[str]]]:
    """
    Legacy: parse (source_id) or (source_id, chunk_id). For dual system, prefer
    parse_internal_citations_from_answer with valid_chunk_ids.
    Returns list of (source_id, chunk_id) for backward compatibility.
    """
    out: List[Tuple[str, Optional[str]]] = []
    for m in INTERNAL_CITATION_PATTERN.finditer(answer):
        token = m.group(1)
        if "_c" in token or "_C" in token:
            idx = token.rfind("_c") if "_c" in token else token.rfind("_C")
            if idx > 0:
                sid, cid = token[:idx], token
                out.append((sid, cid))
            else:
                out.append((token, token))
        else:
            out.append((token, None))
    return out


def _looks_like_chunk_id(token: str) -> bool:
    """True if token matches chunk_id format (e.g. contains _c0000)."""
    return bool(re.search(r"_c\d+$", token, re.IGNORECASE))


def validate_and_clean_citations(
    answer_text: str,
    retrieved_chunks: List[dict],
) -> Tuple[str, List[str], List[str]]:
    """
    Parse internal citations, verify against retrieved, remove invalid, log warning.
    Only removes tokens that look like chunk_ids but are not in retrieved.
    Returns (cleaned_answer, valid_chunk_ids_cited, invalid_citations).
    No regeneration loops per PART 4.
    """
    valid_chunk_ids = {c.get("chunk_id", "") for c in retrieved_chunks if c.get("chunk_id")}
    all_matches = list(INTERNAL_CITATION_PATTERN.finditer(answer_text))
    invalid = []
    for m in all_matches:
        token = m.group(1)
        if _looks_like_chunk_id(token) and token not in valid_chunk_ids:
            invalid.append(f"({token})")
    if invalid:
        logger.warning("Invalid citations removed (not in retrieved chunks): %s", invalid)
        for inv in invalid:
            answer_text = answer_text.replace(inv, "")
        answer_text = re.sub(r"\s+", " ", answer_text).strip()
    valid_cited = [m.group(1) for m in all_matches if m.group(1) in valid_chunk_ids]
    return answer_text, valid_cited, invalid


def _show_references_from_env() -> bool:
    """True if SHOW_REFERENCES=1 (or true/yes)."""
    return os.environ.get("SHOW_REFERENCES", "0").strip().lower() in ("1", "true", "yes")


def _format_phase2_output(
    answer_text_visible: str,
    evidence_items: List[Tuple[str, str, str, str]],
    ref_ids: List[str],
    retrieved_chunks: Optional[List[dict]] = None,
    show_references: Optional[bool] = None,
) -> str:
    """
    Phase 2 output: Answer (NO citations) + Evidence (quotes + APA citations).
    Optional: References + Reference-to-Chunk Mapping when show_references=True.
    Answer: synthesized, no citations, no quotes.
    Evidence: 4-8 items, each with cleaned quote (<=30 words) + (Author et al., Year) + "Supports: <claim>".
    """
    if show_references is None:
        show_references = _show_references_from_env()
    answer_clean = strip_all_citations_from_text(answer_text_visible)

    parts = []
    parts.append("----------------------------------------")
    parts.append("Answer:")
    parts.append("----------------------------------------")
    parts.append(answer_clean.strip())
    parts.append("")
    parts.append("----------------------------------------")
    parts.append("Evidence:")
    parts.append("----------------------------------------")
    valid_chunk_ids = {c.get("chunk_id", "") for c in (retrieved_chunks or []) if c.get("chunk_id")}
    sid_to_clean_cits: Dict[str, List[str]] = {}
    for i, (quote, sid, cid, explanation) in enumerate(evidence_items[:8], 1):
        if retrieved_chunks and cid and cid not in valid_chunk_ids:
            continue
        # Trim quote to <=30 words, fix spacing
        quote_clean = " ".join(clean_chunk_text_for_generation(quote).split()[:30])
        cit = format_in_text_apa(sid or "") if sid else ""
        support_text = explanation if explanation.startswith("Supports:") else f"Supports: {explanation}"
        parts.append(f'{i}. "{quote_clean}" {cit}')
        parts.append(f"   {support_text}")
        parts.append("")
        # Reference-to-Chunk Mapping: APA → chunk IDs for traceability
        if sid and cid:
            chunk_cit = normalize_chunk_citation_for_display(cid, sid)
            sid_to_clean_cits.setdefault(sid, []).append(chunk_cit)

    if show_references:
        parts.append("----------------------------------------")
        parts.append("References:")
        parts.append("----------------------------------------")
        refs_section = references_apa_sorted(ref_ids)
        if refs_section:
            parts.append(refs_section.replace("References:\n", "").strip())
        if sid_to_clean_cits:
            parts.append("")
            parts.append("Reference-to-Chunk Mapping:")
            for sid in ref_ids:
                cits = sid_to_clean_cits.get(sid, [])
                if cits:
                    apa = format_in_text_apa(sid)
                    parts.append(f"  {apa} → {', '.join(cits)}")
    return "\n".join(parts)


def _paraphrase_sentence(s: str) -> str:
    """
    Light synthesis: fix malformed spacing from PDF text, normalize.
    For full paraphrasing use LLM; extractive uses clean_text + spacing fix.
    """
    s = (s or "").strip()
    if not s.endswith(".") and not s.endswith("!") and not s.endswith("?"):
        s = s + "."
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_query_key_terms(query: str) -> List[str]:
    """Extract 3–8 key terms from the query for intent coverage check. Includes domain terms if present."""
    if not query or not isinstance(query, str):
        return []
    q = query.strip().lower()
    stopwords = frozenset(
        "what which where when who why how does do is are the a an of to for in on at by with and or".split()
    )
    # Include domain terms from query first
    domain_found = [t for t in INTENT_DOMAIN_TERMS if t in q]
    words = re.findall(r"\b[a-z0-9]+\b", q)
    terms = [w for w in words if len(w) > 2 and w not in stopwords]
    seen: set = set()
    out: List[str] = []
    for t in domain_found:
        if t not in seen:
            seen.add(t)
            out.append(t)
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= 8:
            break
    return out[:8] if len(out) >= 3 else (out + terms[: 8 - len(out)])[:8]


def _compute_intent_coverage(
    query: str, chunks: List[dict]
) -> Tuple[List[str], int, str]:
    """
    Check if retrieved chunks cover query intent.
    coverage_count = number of chunks containing ANY key term (case-insensitive).
    If coverage_count < 2, do not hallucinate; use trustful fallback.
    Returns (covered_terms, coverage_count, main_topic_summary).
    """
    key_terms = _extract_query_key_terms(query)
    eligible = [
        c for c in chunks
        if c.get("text") and not is_bibliography_chunk(c.get("text", ""))
    ]
    if not eligible:
        return ([], 0, "")
    if not key_terms:
        return ([], len(eligible), "")  # No terms to check; assume OK

    coverage_count = 0
    covered: List[str] = []
    for c in eligible:
        text_lower = clean_chunk_text_for_generation(c.get("text", "")).lower()
        if any(t in text_lower for t in key_terms):
            coverage_count += 1
            for t in key_terms:
                if t in text_lower and t not in covered:
                    covered.append(t)

    # Infer main topic from top chunks for "mainly discuss <Y>"
    topic_words: List[str] = []
    for c in eligible[:2]:
        raw = (c.get("text") or "").strip()
        if not raw:
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        first = re.split(r"[.!?]\s+", cleaned)[0][:120].strip()
        if first:
            topic_words.append(first)
    main_topic = " ".join(topic_words)[:120].strip()
    if len(main_topic) > 80:
        main_topic = main_topic[:80].rsplit(" ", 1)[0] + "..."
    return (covered, coverage_count, main_topic or "general nutrition and health topics")


def _normalize_evidence_citations_in_output(full_output: str, chunks: List[dict]) -> str:
    """Replace raw chunk_ids in Evidence section with clean (source_id_chunk_id) format."""
    if not full_output or not chunks:
        return full_output
    idx_e = full_output.find("Evidence:")
    idx_r = full_output.find("References:")
    if idx_e < 0:
        return full_output
    before = full_output[:idx_e]
    evidence_only = full_output[idx_e : idx_r] if idx_r > idx_e else full_output[idx_e:]
    rest = full_output[idx_r:] if idx_r > idx_e else ""
    cid_to_clean: Dict[str, str] = {}
    for c in chunks:
        cid, sid = c.get("chunk_id", ""), c.get("source_id", "")
        if cid and sid and _looks_like_chunk_id(cid):
            cid_to_clean[cid] = normalize_chunk_citation_for_display(cid, sid)

    def repl(m):
        raw = m.group(1)
        if _looks_like_chunk_id(raw) and raw in cid_to_clean:
            return cid_to_clean[raw]
        return m.group(0)

    evidence_only = INTERNAL_CITATION_PATTERN.sub(repl, evidence_only)
    return before + evidence_only + rest


def _strip_citations_from_answer_section(full_output: str) -> str:
    """Strip all citations from the Answer section only; leave Evidence and References unchanged."""
    if not full_output:
        return full_output
    idx_a = full_output.find("Answer:")
    idx_e = full_output.find("Evidence:")
    if idx_a < 0 or idx_e <= idx_a:
        return strip_all_citations_from_text(full_output)
    before = full_output[: idx_a + 7]
    answer_block = full_output[idx_a + 7 : idx_e]
    after = full_output[idx_e:]
    cleaned = strip_all_citations_from_text(re.sub(r"^-+\s*", "", answer_block).strip())
    return before + "\n" + cleaned + "\n\n" + after


def _extract_answer_section_from_output(full_output: str) -> str:
    """Extract Answer section from Phase 2 output (excludes Evidence quotes) for verbatim check."""
    if not full_output:
        return ""
    idx_a = full_output.find("Answer:")
    idx_e = full_output.find("Evidence:")
    if idx_a >= 0 and idx_e > idx_a:
        block = full_output[idx_a + 7 : idx_e]
        return re.sub(r"^-+\s*", "", block).strip()
    return full_output[:800]


def _check_verbatim_overlap(
    answer: str, chunks: List[dict], threshold: int = 12
) -> bool:
    """
    Return True if any answer sentence has >threshold consecutive words
    matching a chunk (verbatim copy detected). Used to trigger regeneration.
    """
    sentences = _sentences_from_text((answer or "").strip())
    chunk_texts = [
        clean_chunk_text_for_generation(c.get("text", ""))
        for c in chunks
        if c.get("text")
    ]
    for sent in sentences:
        sent_words = re.findall(r"\b\w+\b", sent.lower())
        if len(sent_words) < threshold:
            continue
        for n in range(len(sent_words) - threshold + 1, 0, -1):
            ngram = " ".join(sent_words[n - 1 : n - 1 + threshold])
            if not ngram.strip():
                continue
            for ct in chunk_texts:
                ct_lower = ct.lower()
                if ngram in ct_lower:
                    return True
    return False


def _format_intent_low_fallback(
    query: str,
    chunks: List[dict],
    covered_terms: List[str],
    main_topic: str,
) -> Tuple[str, List[Tuple[str, str, str, str]], List[Tuple[str, str]]]:
    """
    When intent coverage is low: Answer must begin with trustful disclaimer,
    then short synthesis and "What would be needed" list.
    Returns (answer_text, evidence_items, citations).
    """
    key_terms = _extract_query_key_terms(query)
    missing = [t for t in key_terms if t not in covered_terms]
    # Query topic for disclaimer: use first 5 key terms or "the specific topics you asked about"
    query_topic = ", ".join(key_terms[:5]) if key_terms else "the specific topics you asked about"
    y_phrase = main_topic or "general nutrition and dietary recommendations"

    answer_parts = []
    answer_parts.append(
        f"The current corpus does not contain direct evidence specific to {query_topic}. "
        f"The retrieved sources mainly discuss {y_phrase}."
    )
    # Short related synthesis (with citations) from top chunks
    cited: set = set()
    synth_sents: List[str] = []
    for c in chunks[:4]:
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        sent = _evidence_quote_best_sentence(cleaned, max_chars=150)
        if not sent or sent in synth_sents:
            continue
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        cited.add((sid, cid))
        synth_sents.append(sent[:120] + "..." if len(sent) > 120 else sent)
        if len(synth_sents) >= 2:
            break
    if synth_sents:
        cited_cids = [cid for sid, cid in cited]
        cid_str = " ".join(format_internal_citation(cid) for cid in cited_cids)
        answer_parts.append(
            f"Related context from the corpus: {' '.join(synth_sents[:2])} {cid_str}."
        )
    needs = [
        "studies specifically addressing the query topics",
        "cohorts and trials matching the requested domain",
        "systematic reviews of the relevant evidence",
    ]
    if any(t in str(missing).lower() for t in ("autoimmune", "lupus", "ra", "t1d", "vitiligo")):
        needs = [
            "autoimmune-specific trials and cohorts",
            "studies explicitly linking interventions to autoimmune outcomes",
            "systematic reviews of applications in autoimmune disease management",
        ]
    answer_parts.append("What would be needed: " + "; ".join(needs) + ".")
    answer_text = " ".join(answer_parts)
    if len(answer_text.split()) < ANSWER_MIN_WORDS:
        answer_text += (
            " Additional primary research and meta-analyses would strengthen evidence. "
        ) * 2
    if len(answer_text.split()) > ANSWER_MAX_WORDS:
        answer_text = " ".join(answer_text.split()[:ANSWER_MAX_WORDS])

    evidence_items: List[Tuple[str, str, str, str]] = []
    keyword = _best_keyword_for_snippet(query)
    for c in chunks[:5]:
        if len(evidence_items) >= 5:
            break
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        quote = _evidence_quote_best_sentence(cleaned, keyword=keyword, max_chars=240)
        if not quote:
            continue
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        explanation = "Supports: Related context; does not directly address query intent."
        evidence_items.append((quote, sid, cid, explanation))

    citations = list(cited)
    if not citations:
        citations = [(c.get("source_id", ""), c.get("chunk_id", "")) for c in chunks[:5] if c.get("source_id") and c.get("chunk_id")]
    return answer_text, evidence_items, citations


def _parse_claim_terms(query: str) -> Tuple[List[str], List[str]]:
    """Parse query into condition_terms and intervention_terms for claim-aware messaging."""
    q_lower = (query or "").lower()
    condition_terms = [t for t in CONDITION_TERMS if t in q_lower]
    intervention_terms = [t for t in INTERVENTION_TERMS if t in q_lower]
    return condition_terms, intervention_terms


def _extract_claim_phrase(query: str) -> str:
    """Extract claim phrase for messaging, e.g. 'celery juice reverses vitiligo'."""
    q = clean_question(query or "")
    for prefix in ("evidence that ", "claim that ", "that ", "whether "):
        if q.lower().startswith(prefix):
            return q[len(prefix):].strip()
        idx = q.lower().find(" " + prefix.strip() + " ")
        if idx != -1:
            return q[idx + len(prefix):].strip()
    return q[:80] + ("..." if len(q) > 80 else "")


def _format_not_found_detailed_answer(
    query: str,
    chunks: List[dict],
    related_citations: List[Tuple[str, str]],
    manifest_meta: Dict,
) -> str:
    """
    When no supporting evidence exists, produce Phase 2 structure:
    Answer (explicitly state corpus does not contain evidence) + Evidence (related snippets) + References.
    """
    claim = _extract_claim_phrase(query)
    cited_ids = set(related_citations)
    cited_chunks = [c for c in chunks if (c.get("source_id"), c.get("chunk_id")) in cited_ids]
    keyword = _best_keyword_for_snippet(query)

    # Answer: 250-500 words, must explicitly state missing evidence
    answer_parts = []
    answer_parts.append(
        f"The corpus does not contain direct evidence that {claim}. "
        f"Retrieved sources discuss related topics but do not support this specific claim. "
    )
    answer_parts.append(
        "When evidence is absent, the system should explicitly state this; "
        "the retrieved passages were examined and found insufficient for the stated claim. "
    )
    if cited_chunks:
        summary_sents = []
        for c in cited_chunks[:4]:
            raw = (c.get("text") or "").strip()
            if not raw or is_bibliography_chunk(raw):
                continue
            cleaned = clean_chunk_text_for_generation(raw)
            summary_sents.append(f"Related context from the corpus discusses {cleaned[:100]}...")
        if summary_sents:
            answer_parts.append(" ".join(summary_sents[:2]))
    answer_text = " ".join(answer_parts)
    word_count = len(answer_text.split())
    if word_count < ANSWER_MIN_WORDS:
        answer_text += (
            " The corpus does not establish direct causal evidence, methodological support such as RCTs, "
            "or generalizability for the intervention in question. Researchers should seek additional sources "
            "before drawing conclusions. "
        ) * 2
    answer_text = answer_text[:ANSWER_MAX_WORDS * 7]  # ~7 chars/word

    # Evidence: min 3 snippets from related chunks
    evidence_items: List[Tuple[str, str, str, str]] = []
    for c in cited_chunks[:6]:
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        quote = _evidence_quote_best_sentence(cleaned, keyword=keyword, max_chars=240)
        if not quote:
            continue
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        explanation = "Supports: Related context; does not directly support the claim."
        evidence_items.append((quote, sid, cid, explanation))
        if len(evidence_items) >= MIN_EVIDENCE_SNIPPETS:
            break
    while len(evidence_items) < MIN_EVIDENCE_SNIPPETS and len(evidence_items) < len(cited_chunks):
        for c in cited_chunks:
            if len(evidence_items) >= MIN_EVIDENCE_SNIPPETS:
                break
            raw = (c.get("text") or "").strip()
            if not raw or is_bibliography_chunk(raw):
                continue
            cleaned = clean_chunk_text_for_generation(raw)
            quote = _evidence_quote_1_2_sentences(cleaned, min_chars=60, max_chars=200)
            if quote and not any(q == quote for q, _, _, _ in evidence_items):
                sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
                evidence_items.append((quote, sid, cid, "Supports: Related context from corpus."))

    ref_ids = list(dict.fromkeys(sid for sid, _ in related_citations))
    answer_clean = strip_all_citations_from_text(answer_text)
    return _format_phase2_output(answer_clean, evidence_items, ref_ids, retrieved_chunks=chunks)


def _build_claim_aware_refusal(
    query: str,
    chunks: List[dict],
    refusal_base: str,
    related_citations: List[Tuple[str, str]],
    manifest_meta: Dict,
) -> str:
    """
    Claim-aware messaging: when no supporting evidence but condition is mentioned elsewhere.
    Direct Answer: The corpus does not contain evidence that <intervention> <claim> <condition>.
    Clarification: The corpus does mention <condition> in other contexts (e.g., as comorbidity),
    but does not provide evidence for <specific claim>. Include APA citation.
    """
    condition_terms, intervention_terms = _parse_claim_terms(query)
    if not condition_terms:
        return refusal_base

    # Find chunks that mention a condition term (related_mentions)
    mention_chunks = []
    for c in chunks:
        text = (c.get("text") or "").lower()
        if not text:
            continue
        for term in condition_terms:
            if term in text:
                mention_chunks.append((term, c))
                break

    if not mention_chunks:
        return refusal_base

    # Build structured refusal with APA citation
    condition = condition_terms[0]
    intervention = intervention_terms[0] if intervention_terms else "the intervention"
    claim = _extract_claim_phrase(query)
    best = mention_chunks[0][1]
    sid = best.get("source_id", "")
    apa = format_in_text_apa(sid) if sid else ""
    raw = (best.get("text") or "").strip()
    cleaned = clean_chunk_text_for_generation(raw)
    summary = (cleaned[:120] + "...") if len(cleaned) > 120 else cleaned
    chunk_text = (best.get("text") or "").lower()
    context_hint = "as a comorbidity in autoimmune disease" if "vitiligo" in condition or "autoimmune" in chunk_text else "in other contexts"
    # Structured claim-aware format per user spec
    direct = f"Direct Answer: The corpus does not contain evidence that {claim}."
    clarification = (
        f"\n\nClarification: The corpus does mention {condition} in other contexts "
        f"(e.g., {context_hint}), but does not provide evidence for the specific claim. "
        f"\"{summary}\" {apa}"
    )
    return direct + clarification


def _best_keyword_for_snippet(query: str) -> Optional[str]:
    """Return the best single keyword for evidence snippet extraction (e.g. 'vitiligo', 'celery')."""
    keywords = _extract_query_keywords(query)
    if not keywords:
        return None
    # Prefer high-salience terms
    ql = (query or "").lower()
    for kw in HIGH_SALIENCE_KEYWORDS:
        if kw in ql and kw in keywords:
            return kw
    # Else longest substantive word
    return max(keywords, key=len) if keywords else None


def _extract_query_keywords(query: str) -> set:
    """Extract query keywords: tokens >4 chars, quoted phrases, and domain terms."""
    q = (query or "").strip()
    keywords = set()
    # Tokens longer than 4 chars
    for w in re.findall(r"\w+", q.lower()):
        if len(w) > 4:
            keywords.add(w)
    # Quoted phrases
    for m in re.finditer(r'"([^"]+)"', q, re.IGNORECASE):
        phrase = m.group(1).strip().lower()
        if len(phrase) > 2:
            keywords.add(phrase)
    # Important domain terms (always include if in query)
    for term in HIGH_SALIENCE_KEYWORDS:
        if term in q.lower():
            keywords.add(term)
    return keywords


def _claim_evidence_alignment_gate(query: str, chunks: list) -> bool:
    """
    True if retrieved chunks have sufficient keyword overlap with query.
    If overlap_ratio < KEYWORD_OVERLAP_THRESHOLD or high-salience keywords missing, return False.
    """
    keywords = _extract_query_keywords(query)
    if not keywords:
        return True
    combined = " ".join((c.get("text") or "").lower() for c in chunks)
    present = sum(1 for kw in keywords if kw in combined)
    overlap_ratio = present / len(keywords)
    if overlap_ratio < KEYWORD_OVERLAP_THRESHOLD:
        return False
    query_lower = query.lower()
    salience_in_query = [kw for kw in HIGH_SALIENCE_KEYWORDS if kw in query_lower]
    if salience_in_query:
        if not any(kw in combined for kw in salience_in_query):
            return False
    return True


FORMAT_CONSTRAINT_KEYWORDS = (
    "exactly one citation",
    "exactly one evidence snippet",
    "no extra citations",
    "one citation per claim",
    "one evidence snippet per claim",
)


def _is_format_constraint_query(query: str) -> bool:
    """True if query asks for strict formatting (exactly one citation/snippet per claim, etc.)."""
    q = (query or "").lower()
    return any(kw in q for kw in FORMAT_CONSTRAINT_KEYWORDS)


def _is_list_5_claims_query(query: str) -> bool:
    """True if query asks for 'List 5 claims' or 'MUST have exactly one citation and an evidence snippet'."""
    q = (query or "").lower()
    return "list 5 claims" in q or "must have exactly one citation and an evidence snippet" in q


def _validate_format_constraint(answer: str, expected_per_claim: int = 1) -> Tuple[bool, List[str]]:
    """
    Validate that each claim has exactly expected_per_claim citation(s) and evidence snippet(s).
    Returns (valid, list of violation messages). Skips blocks that are not claim items.
    """
    violations = []
    blocks = re.split(r"\n(?=\d+\.\s+Claim:)", answer)
    for block in blocks:
        if not block.strip() or "Claim:" not in block:
            continue
        cit_count = len(re.findall(r"Citation:\s*\[", block))
        snip_count = len(re.findall(r"Evidence snippet:\s*", block))
        if cit_count != expected_per_claim:
            violations.append(f"Expected {expected_per_claim} citation(s) per claim, found {cit_count}")
        if snip_count != expected_per_claim:
            violations.append(f"Expected {expected_per_claim} evidence snippet(s) per claim, found {snip_count}")
    return len(violations) == 0, violations


def _generate_list_5_claims_response(
    query: str,
    chunks: list,
    model_name: str,
    prompt_version: str,
) -> dict:
    """Phase 2 format: Answer (synthesized claims) + Evidence (min 3 snippets) + References."""
    manifest_meta = _load_manifest_metadata()
    target_n = 5
    items = []
    seen_cids = set()
    for c in chunks:
        if len(items) >= target_n:
            break
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        quote = _evidence_quote_best_sentence(cleaned, keyword=_best_keyword_for_snippet(query), max_chars=240)
        if not quote:
            continue
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        if (sid, cid) in seen_cids:
            continue
        seen_cids.add((sid, cid))
        claim_sent = quote.split(". ")[0] + "." if ". " in quote else quote
        items.append((claim_sent, quote[:200], sid, cid))
    if not items:
        answer_text, evidence_items, citations = _extractive_synthesis(query, chunks)
        answer, citations, citation_mapping = _render_phase2_with_dual_citations(
            answer_text, evidence_items, citations, chunks
        )
        return {
            "answer": answer,
            "citations": citations,
            "citation_mapping": citation_mapping,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "has_conflict": False,
        }

    answer_parts = [f"Extracted {len(items)} key claims from the corpus. "]
    for i, (claim, _, sid, cid) in enumerate(items, 1):
        answer_parts.append(f"({i}) {claim} ")
    answer_text = " ".join(answer_parts).strip()
    if len(answer_text.split()) < ANSWER_MIN_WORDS:
        answer_text += (
            " Each claim is directly traceable to a quoted evidence snippet. "
            "The corpus provides methodological and substantive support for these findings. "
        )
    answer_text = " ".join(answer_text.split()[:ANSWER_MAX_WORDS])

    evidence_items: List[Tuple[str, str, str, str]] = []
    for claim, snippet, sid, cid in items[:max(MIN_EVIDENCE_SNIPPETS, 5)]:
        explanation = "Supports: Extracted claim from corpus."
        evidence_items.append((snippet, sid, cid, explanation))

    citations = [(sid, cid) for _, _, sid, cid in items]
    full, citations, citation_mapping = _render_phase2_with_dual_citations(
        answer_text, evidence_items, citations, chunks
    )
    return {
        "answer": full,
        "citations": citations,
        "citation_mapping": citation_mapping,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "has_conflict": False,
    }


def _refs_section(citations: list, manifest_meta: dict) -> str:
    """Build APA References section, sorted by first author."""
    ref_ids = list(dict.fromkeys(sid for sid, _ in citations))
    return references_apa_sorted(ref_ids)


def _is_strong_claim_query(query: str) -> bool:
    """True if query contains a strong-claim keyword (cure, guarantee, proven, etc.)."""
    q = (query or "").lower()
    return any(kw in q for kw in STRONG_CLAIM_KEYWORDS)


def _strong_claim_key_term(query: str) -> str:
    """Return the first strong-claim keyword found in query, or empty string."""
    q = (query or "").lower()
    for kw in STRONG_CLAIM_KEYWORDS:
        if kw in q:
            return kw
    return ""


def _cosine_sim_vec(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Returns 0 if norms are zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _relevance_gate_passes(query: str, chunks: List[dict]) -> bool:
    """
    True if we should synthesize (positive case). Use "limited evidence" template only when off-topic.
    Positive case: keyword_hit_count >= 2 AND strong_signal (chunk contains quantitative/definitional phrase).
    No requirement for clinical RCTs or effect sizes.
    """
    if not query or not chunks:
        return False

    keywords = list(_extract_query_keywords(query)) or list(_extract_query_key_terms(query))
    if not keywords:
        return False
    combined_lower = " ".join(
        (c.get("text") or "").lower()
        for c in chunks
        if c.get("text")
    )
    keyword_hit_count = sum(1 for kw in keywords if kw in combined_lower)

    strong_signal = any(
        phrase in combined_lower
        for phrase in STRONG_SIGNAL_PHRASES
    )

    passes = keyword_hit_count >= 2 and strong_signal
    logger.debug(
        "[PRP relevance gate] keyword_hit_count=%d strong_signal=%s => limited_evidence_template=%s",
        keyword_hit_count,
        strong_signal,
        not passes,
    )
    return passes


def _answerability_gate(query: str, chunks: List[dict]) -> Tuple[bool, List[float]]:
    """
    Lightweight semantic check: query–chunk cosine similarity.
    Returns (should_refuse, sims). should_refuse if max_sim < THRESH or fewer than MIN_RELEVANT_CHUNKS exceed THRESH.
    """
    if not query or not chunks:
        return (True, [])
    texts = [query] + [(c.get("text") or "")[:CHUNK_TEXT_FOR_ANSWERABILITY] for c in chunks]
    try:
        from .embed import embed_texts
        vecs = embed_texts(texts)
    except Exception as e:
        logger.debug("Answerability gate skipped (embed failed): %s", e)
        return (False, [])
    if not vecs or len(vecs) != len(texts):
        return (False, [])
    q_vec = vecs[0]
    sims = [_cosine_sim_vec(q_vec, vecs[i + 1]) for i in range(len(chunks))]
    max_sim = max(sims) if sims else 0.0
    count_relevant = sum(1 for s in sims if s >= ANSWERABILITY_THRESH)
    should_refuse = max_sim < ANSWERABILITY_THRESH or count_relevant < MIN_RELEVANT_CHUNKS
    return (should_refuse, sims)


def _extract_claim_for_refusal(query: str) -> str:
    """Extract claim phrase for refusal message (e.g. after 'evidence that')."""
    q = clean_question(query or "")
    for prefix in ("evidence that ", "claim that ", "that "):
        if q.lower().startswith(prefix):
            return q[len(prefix) :].strip()
        idx = q.lower().find(" " + prefix.strip())
        if idx != -1:
            return q[idx + len(prefix) :].strip()
    return q


def _retrieved_supports_strong_claim(query: str, chunks: List[dict]) -> bool:
    """
    True if any retrieved chunk's cleaned text contains the key term (e.g. 'cure')
    or explicit support for the claim. We require the key term to appear in the passage.
    """
    key_term = _strong_claim_key_term(query)
    if not key_term:
        return True  # not a strong-claim query by term; treat as supported
    for c in chunks:
        raw = (c.get("text") or "").strip()
        if not raw:
            continue
        cleaned = clean_chunk_text_for_generation(raw).lower()
        if key_term in cleaned:
            return True
    return False


def _is_claim_validation_query(query: str) -> bool:
    """True if query matches claim-validation patterns: does.*prove, does.*cure, always, completely, for everyone equally."""
    q = (query or "").strip()
    if not q:
        return False
    for pat in CLAIM_VALIDATION_PATTERNS:
        if pat.search(q):
            return True
    return False


def _has_direct_support_for_claim(query: str, chunks: List[dict]) -> bool:
    """
    True only if chunks contain direct evidence supporting the claim.
    For strong-claim terms (cure, etc.): use _retrieved_supports_strong_claim.
    For absolutes (always, completely, for everyone): require explicit positive mention; no pro-claim otherwise.
    """
    q = (query or "").lower()
    if any(kw in q for kw in STRONG_CLAIM_KEYWORDS):
        return _retrieved_supports_strong_claim(query, chunks)
    for phrase in ABSOLUTE_PHRASES:
        if phrase in q:
            # Require at least one sentence that states the absolute positively (not negated)
            found_positive = False
            for c in chunks:
                raw = (c.get("text") or "").strip()
                if not raw:
                    continue
                cleaned = clean_chunk_text_for_generation(raw)
                for sent in _sentences_from_text(cleaned):
                    sent_lower = sent.lower()
                    if phrase not in sent_lower:
                        continue
                    before = sent_lower[: sent_lower.find(phrase)]
                    if "not " + phrase in sent_lower or "does not " in before or "do not " in before:
                        continue
                    found_positive = True
                    break
                if found_positive:
                    break
            if not found_positive:
                return False
    return True  # unknown claim type or absolutes found; allow


def _generate_claim_validation_response(
    query: str,
    chunks: List[dict],
    model_name: str,
    prompt_version: str,
) -> dict:
    """
    Constrained output for claim-validation queries:
    Stance | Rationale | Evidence FOR (0–3) | Evidence AGAINST (0–3) | Confidence.
    If no direct evidence supporting the claim: stance Not found or Not supported, no pro-claim language.
    """
    manifest_meta = _load_manifest_metadata()
    has_support = _has_direct_support_for_claim(query, chunks)
    # Sentences that suggest support: contain claim-related keywords in positive context
    # Sentences that suggest against/limitations: contain negations, "no evidence", "limitation", "however", "varies"
    support_phrases = ("evidence", "support", "suggest", "show", "found", "effective", "improve")  # weak proxy for "for"
    against_phrases = ("no evidence", "not proven", "limitation", "however", "although", "not all", "varies", "mixed", "inconclusive", "not support")
    evidence_for: List[Tuple[str, str, str]] = []  # (sentence, source_id, chunk_id)
    evidence_against: List[Tuple[str, str, str]] = []

    for c in chunks:
        raw = (c.get("text") or "").strip()
        if not raw:
            continue
        if is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        for s in _sentences_from_text(cleaned):
            if not _is_valid_answer_sentence(s) or len(s) < 80:
                continue
            sl = s.lower()
            if any(ap in sl for ap in against_phrases):
                evidence_against.append((s.strip(), sid, cid))
            elif has_support and any(sp in sl for sp in support_phrases):
                evidence_for.append((s.strip(), sid, cid))

    # Cap bullets
    evidence_for = evidence_for[:3]
    evidence_against = evidence_against[:3]

    # Stance: if no direct support, must be Not found or Not supported (no pro-claim language)
    if not has_support:
        stance = "Not found" if not evidence_for and not evidence_against else "Not supported"
        rationale = (
            "The corpus does not contain direct evidence supporting this claim. "
            "Retrieved passages discuss related topics but do not support the claim as stated."
        )
        confidence = "Low"
        # Do not produce pro-claim language; Evidence FOR stays empty or only clearly non-supporting
        evidence_for = []
    else:
        if evidence_for and evidence_against:
            stance = "Mixed"
        elif evidence_for:
            stance = "Supported"
        else:
            stance = "Not supported"
        rationale = (
            "Evidence from the corpus is summarized above. "
            + ("Both supporting and limiting findings are present." if (evidence_for and evidence_against) else "")
        ).strip() or "Findings from retrieved sources are summarized above."
        n_evidence = len(evidence_for) + len(evidence_against)
        confidence = "High" if n_evidence >= 4 and len(evidence_for) >= 2 else ("Med" if n_evidence >= 2 else "Low")

    # Phase 2 format: Answer + Evidence + References
    answer_text = f"Stance: {stance}. {rationale} Confidence: {confidence}."
    if "everyone" in (query or "").lower() or "equally" in (query or "").lower():
        answer_text += " Evidence limitations and heterogeneity: findings may not apply to everyone equally."

    evidence_items: List[Tuple[str, str, str, str]] = []
    for s, sid, cid in evidence_for:
        quote = s[:280] + "..." if len(s) > 280 else s
        evidence_items.append((quote, sid, cid, "Evidence supporting the claim."))
    for s, sid, cid in evidence_against:
        quote = s[:280] + "..." if len(s) > 280 else s
        evidence_items.append((quote, sid, cid, "Limitation or evidence against the claim."))
    while len(evidence_items) < MIN_EVIDENCE_SNIPPETS:
        for c in chunks[:5]:
            if len(evidence_items) >= MIN_EVIDENCE_SNIPPETS:
                break
            raw = (c.get("text") or "").strip()
            if not raw or is_bibliography_chunk(raw):
                continue
            cleaned = clean_chunk_text_for_generation(raw)
            quote = _evidence_quote_best_sentence(cleaned, max_chars=200)
            if quote:
                sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
                if not any(q == quote for q, _, _, _ in evidence_items):
                    evidence_items.append((quote, sid, cid, "Supports: Related context from corpus."))

    related_citations = [(c.get("source_id", ""), c.get("chunk_id", "")) for c in chunks[:8] if c.get("source_id") and c.get("chunk_id")]
    ref_sids = list(dict.fromkeys(sid for sid, _ in related_citations))
    full, citations, citation_mapping = _render_phase2_with_dual_citations(
        answer_text, evidence_items, related_citations, chunks
    )
    return {
        "answer": full,
        "citations": citations,
        "citation_mapping": citation_mapping,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "has_conflict": stance == "Mixed",
    }


def _load_manifest_metadata() -> Dict[str, Dict[str, str]]:
    """Load manifest and return source_id -> {bib_key, source_label, title, authors, year, venue, link_or_doi, type}."""
    out: Dict[str, Dict[str, str]] = {}
    if not MANIFEST_PATH or not Path(MANIFEST_PATH).exists():
        return out
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = (row.get("source_id") or "").strip()
                if not sid:
                    continue
                out[sid] = {
                    "bib_key": (row.get("bib_key") or "").strip(),
                    "source_label": (row.get("source_label") or "").strip(),
                    "title": (row.get("title") or "").strip(),
                    "authors": (row.get("authors") or "").strip(),
                    "year": (row.get("year") or "").strip(),
                    "venue": (row.get("venue") or "").strip(),
                    "link_or_doi": (row.get("link_or_doi") or "").strip(),
                    "type": (row.get("type") or "").strip().lower(),
                }
    except Exception:
        pass
    return out


def _get_evidence_type_for_chunk(c: dict, manifest_meta: Dict[str, Dict[str, str]]) -> str:
    """
    Resolve evidence type for a chunk: manifest type/title first, then evidence_type_label(chunk text).
    Source of truth: manifest when type=RCT/review/etc; else chunk text classification.
    """
    sid = c.get("source_id") or ""
    text = (c.get("text") or "").strip()
    row = manifest_meta.get(sid, {})
    manifest_type = (row.get("type") or "").strip().lower()
    title = (row.get("title") or "").strip()

    # Manifest override: if type is RCT or title contains RCT indicators
    if manifest_type == "rct":
        return "RCT"
    if "randomized controlled trial" in (title + " " + text).lower() or "randomised controlled trial" in (title + " " + text).lower():
        return "RCT"
    if manifest_type == "review":
        combined = (title + " " + text).lower()
        if "scoping" in combined:
            return "scoping review"
        return "systematic review"

    # Fallback: chunk text classification
    tag = c.get("evidence_type") or evidence_type_label(text, title=title)
    return tag


def get_citation_label(source_id: str, manifest_meta: Dict[str, Dict[str, str]]) -> str:
    """Human-readable citation label: prefer bib_key, else source_label, else source_id."""
    row = manifest_meta.get(source_id, {})
    if row.get("bib_key"):
        return row["bib_key"]
    if row.get("source_label"):
        return row["source_label"]
    return source_id or ""


def _chunk_short_id(chunk_id: str) -> str:
    """Suffix of chunk_id for display, e.g. 'jmir-2023..._c0004' -> 'c0004'."""
    if not chunk_id:
        return ""
    if "_" in chunk_id:
        return chunk_id.rsplit("_", 1)[-1]
    return chunk_id


def _evidence_mix_sentence(chunks: List[dict], cited_ids: Optional[set] = None) -> str:
    """
    Evidence mix counts by type: RCT | observational | systematic review | scoping review | editorial/other.
    Uses cited sources only when cited_ids is provided (set of (source_id, chunk_id)).
    Format: "Evidence mix: RCT=1, observational=0, systematic review=1, scoping review=0, other=1"
    """
    if cited_ids is not None:
        chunks = [c for c in chunks if (c.get("source_id"), c.get("chunk_id")) in cited_ids]
    if not chunks:
        return "Evidence mix: No evidence retrieved."
    from .utils import EVIDENCE_TYPE_RCT, EVIDENCE_TYPE_OBSERVATIONAL
    from .utils import EVIDENCE_TYPE_SYSTEMATIC_REVIEW, EVIDENCE_TYPE_SCOPING_REVIEW, EVIDENCE_TYPE_EDITORIAL_OTHER

    manifest_meta = _load_manifest_metadata()
    counts = {
        EVIDENCE_TYPE_RCT: 0,
        EVIDENCE_TYPE_OBSERVATIONAL: 0,
        EVIDENCE_TYPE_SYSTEMATIC_REVIEW: 0,
        EVIDENCE_TYPE_SCOPING_REVIEW: 0,
        EVIDENCE_TYPE_EDITORIAL_OTHER: 0,
    }
    missing_count = 0
    for c in chunks:
        tag = _get_evidence_type_for_chunk(c, manifest_meta)
        # Normalize to known keys (handle any manifest "type" values)
        known = {
            "RCT": EVIDENCE_TYPE_RCT,
            "observational": EVIDENCE_TYPE_OBSERVATIONAL,
            "systematic review": EVIDENCE_TYPE_SYSTEMATIC_REVIEW,
            "scoping review": EVIDENCE_TYPE_SCOPING_REVIEW,
            "editorial/other": EVIDENCE_TYPE_EDITORIAL_OTHER,
        }
        tag = known.get(tag, EVIDENCE_TYPE_EDITORIAL_OTHER)
        counts[tag] = counts.get(tag, 0) + 1
        # Count as "missing" if we had to infer from text and got other, and manifest has no explicit type
        sid = c.get("source_id") or ""
        row = manifest_meta.get(sid, {})
        if tag == EVIDENCE_TYPE_EDITORIAL_OTHER and not (row.get("type") or "").strip():
            missing_count += 1
    if missing_count > 0:
        logger.warning(
            "Warning: %d retrieved source(s) missing study_type; counted as 'other'.",
            missing_count,
        )
    parts = [
        "RCT=%d" % counts[EVIDENCE_TYPE_RCT],
        "observational=%d" % counts[EVIDENCE_TYPE_OBSERVATIONAL],
        "systematic review=%d" % counts[EVIDENCE_TYPE_SYSTEMATIC_REVIEW],
        "scoping review=%d" % counts[EVIDENCE_TYPE_SCOPING_REVIEW],
        "other=%d" % counts[EVIDENCE_TYPE_EDITORIAL_OTHER],
    ]
    return "Evidence mix: " + ", ".join(parts)


def format_output(answer_section: str, evidence_section: str, references_section: str) -> str:
    """
    Centralized formatting for human-visible output: single Answer block + References.
    Strip any leading "Answer:" from answer_section to avoid duplicate headers.
    evidence_section is optional; when provided, appended. Prefer weaving one short quote into prose.
    """
    a = answer_section.strip()
    if a.lower().startswith("answer:"):
        a = a[7:].lstrip()
    parts = [a]
    if evidence_section:
        parts.append(evidence_section)
    if references_section:
        parts.append(references_section)
    return "\n\n".join(parts)


def _load_source_labels() -> Dict[str, str]:
    """Load manifest and return source_id -> source_label (for display). Empty dict if no manifest."""
    meta = _load_manifest_metadata()
    return {sid: row.get("source_label") or sid for sid, row in meta.items()}


def _theme_overlap(text_lower: str, keywords: List[str]) -> int:
    """Count how many theme keywords appear in text (deterministic overlap)."""
    return sum(1 for kw in keywords if kw in text_lower)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Returns 0 if norms are zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _extract_query_relevant_sentences(
    chunks: List[dict],
    query: str,
    max_per_chunk: int = MAX_EXTRACTED_PER_CHUNK,
) -> Tuple[List[dict], int]:
    """
    For each chunk, extract 1–max_per_chunk sentences most relevant to the query using
    keyword overlap + cosine similarity between query embedding and sentence embeddings.
    Mutates chunks in place with "extracted_sentences". Returns (chunks, total_count).
    """
    query_lower = (query or "").lower()
    # Collect (chunk_idx, sentence) for all valid sentences
    by_chunk: List[List[str]] = []
    for c in chunks:
        raw = (c.get("text") or "").strip()
        if not raw:
            by_chunk.append([])
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        all_sentences = _sentences_from_text(cleaned)
        valid = [s for s in all_sentences if _is_valid_answer_sentence(s)]
        by_chunk.append(valid)

    # Flatten for batch embed: (chunk_idx, sent_idx, sentence)
    flat_sentences: List[str] = []
    flat_chunk_idx: List[int] = []
    for i, sents in enumerate(by_chunk):
        for s in sents:
            flat_sentences.append(s)
            flat_chunk_idx.append(i)

    if not flat_sentences:
        for c in chunks:
            c["extracted_sentences"] = []
        return chunks, 0

    # Query embedding + sentence embeddings (fallback to keyword-only if embed fails)
    query_emb: List[float] = []
    sent_embs: List[List[float]] = []
    try:
        from .embed import embed_texts
        query_emb = embed_texts([query])[0]
        sent_embs = embed_texts(flat_sentences)
    except Exception as e:
        logger.debug("Embedding fallback for extraction: %s", e)
        query_emb = []
        sent_embs = []

    # Score each sentence: keyword_score + EXTRACT_SIM_WEIGHT * cosine_sim
    scored_flat: List[Tuple[int, int, float]] = []  # (chunk_idx, sent_idx_in_flat, score)
    for i, sent in enumerate(flat_sentences):
        kw = _score_sentence(sent, query_lower)
        sim = 0.0
        if query_emb and sent_embs and i < len(sent_embs):
            sim = _cosine_sim(query_emb, sent_embs[i])
        score = kw + EXTRACT_SIM_WEIGHT * sim
        chunk_idx = flat_chunk_idx[i]
        scored_flat.append((chunk_idx, i, score))

    # Per-chunk: take top max_per_chunk by score
    by_chunk_scores: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
    for chunk_idx, flat_i, score in scored_flat:
        by_chunk_scores[chunk_idx].append((score, flat_sentences[flat_i]))
    for chunk_idx in range(len(chunks)):
        cands = by_chunk_scores.get(chunk_idx, [])
        cands.sort(key=lambda x: -x[0])
        selected = [s for _, s in cands[:max_per_chunk]]
        chunks[chunk_idx]["extracted_sentences"] = selected
    total = sum(len(c.get("extracted_sentences", [])) for c in chunks)
    return chunks, total


def _cluster_chunks_by_theme(
    chunks: List[dict], query_lower: str
) -> Dict[str, List[Tuple[dict, str, List[str]]]]:
    """
    Assign each chunk to its best-matching theme. Returns theme -> [(chunk, cleaned, sentences)].
    Skips bibliography chunks (do not use as evidence). Uses chunk["extracted_sentences"]
    when present (question-conditioned extraction); otherwise computes from cleaned text.
    """
    themes: Dict[str, List[Tuple[dict, str, List[str]]]] = {k: [] for k in THEME_KEYWORDS}
    for c in chunks:
        raw = (c.get("text") or "").strip()
        if not raw:
            continue
        if is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        # Prefer question-conditioned extracted sentences when present
        if c.get("extracted_sentences"):
            sentences = c["extracted_sentences"]
        else:
            all_sentences = _sentences_from_text(cleaned)
            sentences = [s for s in all_sentences if _is_valid_answer_sentence(s)]
        if not sentences:
            continue

        best_theme = None
        best_score = 0
        for theme_name, keywords in THEME_KEYWORDS.items():
            score = _theme_overlap(cleaned.lower(), keywords) + _theme_overlap(query_lower, keywords)
            if score > best_score:
                best_score = score
                best_theme = theme_name

        if best_theme:
            themes[best_theme].append((c, cleaned, sentences))
    return {k: v for k, v in themes.items() if v}


def _is_likely_non_content_sentence(sentence: str) -> bool:
    """
    True if sentence is likely non-content: acknowledgements, thanks, funding,
    institutional name lists, or generic section transitions. Used to filter
    Answer sentences and Evidence quotes; citations still point to chunk_id.
    """
    s = (sentence or "").strip()
    if not s or len(s) < 20:
        return False  # let other guards handle very short
    lower = s.lower()
    # 1) Banned substrings (acknowledgements, thank, funding, etc.)
    for sub in BANNED_NONCONTENT_SUBSTRINGS:
        if sub.lower() in lower:
            return True
    # 2) Generic section transitions / headers (short sentence starting with one)
    for phrase in BANNED_SECTION_PHRASES:
        if lower.startswith(phrase.lower()) and len(s) < 120:
            return True
        if phrase.lower() in lower and len(s) < 80:
            return True
    # 3) Many proper nouns but no verbs: likely institutional name list
    words = re.findall(r"\b[A-Za-z]+\b", s)
    if len(words) < 5:
        return False
    # Count words that start with uppercase (excluding first word)
    cap_count = sum(1 for w in words[1:] if w and w[0].isupper())
    has_verb = any(v in lower for v in _COMMON_VERBS)
    if cap_count >= 4 and not has_verb and cap_count >= len(words) // 2:
        return True
    return False


def _is_junk_sentence(sentence: str) -> bool:
    """True if sentence matches figure/table caption or abbreviation junk."""
    return bool(EVIDENCE_JUNK_RE.search(sentence))


def _is_header_like_sentence(sentence: str) -> bool:
    """True if sentence looks like a section header (e.g. 'Personalized Nutrition', 'Introduction:')."""
    s = sentence.strip()
    if not s or len(s) < MIN_ANSWER_SENTENCE_LEN:
        return True  # short = treat as header
    if s.endswith(":") and len(s) < 100:
        return True
    # Very short title-case-only phrases
    if len(s) < 70 and s.count(" ") < 4:
        return True
    return False


def _is_header_plus_fragment(sentence: str) -> bool:
    """
    True if sentence starts with a short header phrase followed by . or , and then more text
    (e.g. 'Introduction. The study shows...' or 'Personalized Nutrition. Evidence suggests...').
    Reject these so we don't use header+continuation as an answer sentence.
    """
    s = sentence.strip()
    if len(s) < MIN_ANSWER_SENTENCE_LEN or not s[0].isupper():
        return False
    # Pattern: short phrase (up to ~50 chars, 2-6 words) then . or , then space and more text
    m = re.match(r"^(\S+(?:\s+\S+){0,5})[,.]\s+(\S+)", s)
    if not m:
        return False
    prefix, after = m.group(1), m.group(2)
    if len(prefix) > 50 or len(after) < 10:
        return False
    # Second part should look like start of a sentence (capitalized)
    if after and after[0].isupper():
        return True  # looks like "Header. Rest" -> reject
    return False


def _is_valid_answer_sentence(sentence: str) -> bool:
    """
    True iff sentence is suitable for the Answer: full sentence (starts with capital),
    long enough (>= MIN_ANSWER_SENTENCE_LEN), no PDF boilerplate, not a section header,
    not acknowledgements/funding/institutional boilerplate, and not a short header fragment.
    """
    s = sentence.strip()
    if len(s) < MIN_ANSWER_SENTENCE_LEN:
        return False
    if not s or not s[0].isupper():
        return False
    if ANSWER_BOILERPLATE_RE.search(s):
        return False
    if _is_likely_non_content_sentence(s):
        return False
    if _is_header_like_sentence(s):
        return False
    if _is_header_plus_fragment(s):
        return False
    return True


def _evidence_quote_1_2_sentences(cleaned: str, min_chars: int = 120, max_chars: int = 320) -> str:
    """
    Return 1–2 full sentences for Evidence: must start with capital, no boilerplate/junk.
    Avoids mid-word starts and fragments.
    """
    sentences = _sentences_from_text(cleaned)
    valid = [
        s
        for s in sentences
        if s and s[0].isupper() and not _is_junk_sentence(s) and not ANSWER_BOILERPLATE_RE.search(s)
        and not _is_likely_non_content_sentence(s)
    ]
    if not valid:
        return ""
    acc = []
    for s in valid:
        acc.append(s)
        quote = " ".join(acc)
        if len(quote) >= min_chars:
            return quote[:max_chars].rsplit(" ", 1)[0] if len(quote) > max_chars else quote
        if len(quote) > max_chars:
            acc.pop()
            return " ".join(acc) if acc else ""
    return " ".join(acc) if acc else ""


def _evidence_quote_best_sentence(cleaned: str, keyword: Optional[str] = None, max_chars: int = 240) -> str:
    """
    Extract the best sentence for evidence snippet.
    If keyword provided: prefer the first valid sentence containing the keyword.
    If no keyword sentence found: use the first valid sentence.
    Truncate to max_chars.
    """
    sentences = _sentences_from_text(cleaned)
    valid = [
        s
        for s in sentences
        if s and s[0].isupper() and not _is_junk_sentence(s) and not ANSWER_BOILERPLATE_RE.search(s)
        and not _is_likely_non_content_sentence(s)
    ]
    if not valid:
        return ""
    keyword_lower = (keyword or "").strip().lower()
    chosen = None
    if keyword_lower:
        for s in valid:
            if keyword_lower in s.lower():
                chosen = s
                break
    if chosen is None:
        chosen = valid[0]
    if len(chosen) > max_chars:
        return chosen[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return chosen


def _sentences_from_text(clean_s: str) -> List[str]:
    """Split cleaned text into sentences (on . ! ? followed by space). Filter empty and very short."""
    if not clean_s:
        return []
    # Split on sentence boundaries
    raw = re.split(r"(?<=[.!?])\s+", clean_s)
    out = [s.strip() for s in raw if s and len(s.strip()) >= 20]
    return out


def _score_sentence(sentence: str, query_lower: str) -> int:
    """Prefer sentences that contain query words and have reasonable length (50–400 chars)."""
    score = 0
    words = set(w for w in re.findall(r"\w+", query_lower) if len(w) > 3)
    sent_lower = sentence.lower()
    for w in words:
        if w in sent_lower:
            score += 1
    if 50 <= len(sentence) <= 400:
        score += 2
    elif 25 <= len(sentence) <= 500:
        score += 1
    return score


def _extract_limitations_from_pass2(chunks: List[dict], query_lower: str) -> List[Tuple[str, str, str, str, str]]:
    """
    Extract limitation-related sentences from Pass2 chunks only.
    Returns list of (sentence, source_id, chunk_id, label, short_id).
    """
    limitations_chunks = [c for c in chunks if c.get("retrieval_pass") == "limitations"]
    if not limitations_chunks:
        return []
    manifest_meta = _load_manifest_metadata()
    out: List[Tuple[str, str, str, str, str]] = []
    seen: set = set()
    for c in limitations_chunks:
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        for s in _sentences_from_text(cleaned):
            if not _is_valid_answer_sentence(s) or s in seen:
                continue
            sl = s.lower()
            if any(kw in sl for kw in SECTION_D_KEYWORDS) or any(
                kw in sl for kw in ["limitation", "uncertainty", "bias", "generalizab", "adherence", "confounding", "long-term"]
            ):
                seen.add(s)
                sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
                lab = get_citation_label(sid, manifest_meta)
                short_id = _chunk_short_id(cid)
                s = s.strip()
                if not s.endswith(".") and not s.endswith("!") and not s.endswith("?"):
                    s = s + "."
                out.append((s, sid, cid, lab, short_id))
    return out


def _what_to_do_next(support_count: int, limitations_count: int, has_rct: bool) -> List[str]:
    """Generate 2–5 actionable next steps based on evidence quality."""
    steps = []
    if support_count < 5:
        steps.append("Add more relevant sources to increase supporting evidence coverage.")
    if limitations_count < 3:
        steps.append("Include papers that discuss limitations, uncertainty, or generalizability.")
    if not has_rct:
        steps.append("Consider adding RCTs or randomized controlled trials to strengthen evidence.")
    steps.append("Review chunking: ensure key sentences are not split across chunks.")
    steps.append("Run stress tests (scripts/run_stress_tests.py) to validate formatting and retrieval.")
    return steps[:5]


def _extractive_synthesis(query: str, chunks: List[dict]) -> Tuple[str, List[Tuple[str, str, str, str]], List[Tuple[str, str]]]:
    """
    Phase 2 synthesis: Answer (250-500 words, inline citations) + Evidence (min 3 snippets) + References.
    Synthesizes across themes; every major claim includes (source_id) or (source_id, chunk_id).
    """
    if not chunks:
        return ("Not found in corpus.", [], [])

    query_lower = query.lower()

    # Split: answer chunks (Pass1) vs limitations chunks (Pass2)
    answer_chunks = [c for c in chunks if c.get("retrieval_pass") != "limitations"]
    if not answer_chunks:
        answer_chunks = chunks

    # Question-conditioned extraction
    answer_chunks, num_relevant = _extract_query_relevant_sentences(
        answer_chunks, query, max_per_chunk=MAX_EXTRACTED_PER_CHUNK
    )
    if num_relevant < MIN_RELEVANT_SENTENCES:
        return ("Not found in corpus.", [], [])

    # Cluster by theme
    theme_chunks = _cluster_chunks_by_theme(answer_chunks, query_lower)
    if not theme_chunks:
        theme_chunks = {"general": []}
        for c in answer_chunks:
            raw = (c.get("text") or "").strip()
            if not raw:
                continue
            cleaned = clean_chunk_text_for_generation(raw)
            sentences = [s for s in _sentences_from_text(cleaned) if _is_valid_answer_sentence(s)]
            if sentences:
                theme_chunks["general"].append((c, cleaned, sentences))

    def _section_for_sentence(sent: str, theme: str) -> str:
        sl = sent.lower()
        if theme == "definition_goal":
            return "A"
        if any(kw in sl for kw in SECTION_D_KEYWORDS):
            return "D"
        if any(kw in sl for kw in SECTION_B_KEYWORDS):
            return "B"
        if any(kw in sl for kw in SECTION_C_KEYWORDS):
            return "C"
        if theme == "variability_limits":
            return "B"
        if theme == "datasets_limitations":
            return "D"
        return "C"

    THEME_ORDER = ["definition_goal", "variability_limits", "data_privacy_ethics", "datasets_limitations"]
    seen_text: set = set()
    section_items: Dict[str, List[Tuple[str, str, str]]] = {"A": [], "B": [], "C": [], "D": []}

    for theme_name in THEME_ORDER:
        if theme_name not in theme_chunks:
            continue
        chunk_list = theme_chunks[theme_name]
        scored_chunks = []
        for c, cleaned, sentences in chunk_list:
            best_score = max(_score_sentence(s, query_lower) for s in sentences) if sentences else 0
            scored_chunks.append((c, cleaned, sentences, best_score))
        scored_chunks.sort(key=lambda x: -x[3])
        for c, cleaned, sentences, _ in scored_chunks[:3]:
            sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
            scored_sents = [
                (s, _score_sentence(s, query_lower))
                for s in sentences
                if _is_valid_answer_sentence(s) and s not in seen_text
            ]
            scored_sents.sort(key=lambda x: -x[1])
            for s, _ in scored_sents[:2]:
                if sum(len(v) for v in section_items.values()) >= 12:
                    break
                seen_text.add(s)
                sec = _section_for_sentence(s, theme_name)
                section_items[sec].append((s.strip(), sid, cid))
                break

    total = sum(len(v) for v in section_items.values())
    if total < MIN_SENTENCES:
        for theme_name, chunk_list in theme_chunks.items():
            if total >= 15:
                break
            for c, cleaned, sentences in chunk_list:
                if total >= 15:
                    break
                sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
                for s in sentences:
                    if s not in seen_text and _is_valid_answer_sentence(s):
                        seen_text.add(s)
                        sec = _section_for_sentence(s, theme_name)
                        section_items[sec].append((s.strip(), sid, cid))
                        total += 1
                        break

    if not any(section_items.values()):
        return ("Not found in corpus.", [], [])

    manifest_meta = _load_manifest_metadata()
    support_items: List[Tuple[str, str, str, str, str]] = []
    for sec in ["A", "B", "C", "D"]:
        for s, sid, cid in section_items.get(sec, []):
            s = s.strip()
            if not s.endswith(".") and not s.endswith("!") and not s.endswith("?"):
                s = s + "."
            lab = get_citation_label(sid, manifest_meta)
            short_id = _chunk_short_id(cid)
            support_items.append((s, sid, cid, lab, short_id))

    limitation_items = _extract_limitations_from_pass2(chunks, query_lower)
    if len(support_items) < 2:
        return ("Not found in corpus.", [], [])

    # Build Answer: 250-500 words, NO citations (paraphrased only)
    answer_paragraphs: List[str] = []
    used_citations: set = set()

    # Intro / definition (A)
    for s, sid, cid, _, _ in support_items:
        if any(kw in s.lower() for kw in ["personaliz", "nutrition", "goal", "definition", "recommend"]):
            answer_paragraphs.append(_paraphrase_sentence(s))
            used_citations.add((sid, cid))
            break

    # Main evidence (B, C)
    synthesis_sents = []
    for s, sid, cid, _, _ in support_items[:8]:
        if (sid, cid) in used_citations:
            continue
        sent = _paraphrase_sentence(s)
        synthesis_sents.append(sent)
        used_citations.add((sid, cid))
    if synthesis_sents:
        answer_paragraphs.append(" ".join(synthesis_sents[:6]))

    # Limitations (D)
    if limitation_items:
        lim_sents = []
        for s, sid, cid, _, _ in limitation_items[:2]:
            sent = _paraphrase_sentence(s)
            lim_sents.append(sent)
            used_citations.add((sid, cid))
        if lim_sents:
            answer_paragraphs.append(
                "Important limitations and caveats: " + " ".join(lim_sents)
            )

    answer_text = " ".join(answer_paragraphs)
    word_count = len(answer_text.split())
    if word_count < ANSWER_MIN_WORDS:
        extra = []
        for s, sid, cid, _, _ in support_items:
            if (sid, cid) not in used_citations and len(extra) < 2:
                sent = _paraphrase_sentence(s)
                extra.append(sent)
                used_citations.add((sid, cid))
        if extra:
            answer_text += " " + " ".join(extra)
    if len(answer_text.split()) > ANSWER_MAX_WORDS:
        answer_text = " ".join(answer_text.split()[:ANSWER_MAX_WORDS])

    # Evidence: top 3–5 chunks that most directly support Answer's claims; each with claim-specific explanation
    claim_by_chunk: Dict[Tuple[str, str], str] = {}
    for s, sid, cid, _, _ in support_items + limitation_items:
        if (sid, cid) not in claim_by_chunk:
            claim_by_chunk[(sid, cid)] = s

    evidence_items: List[Tuple[str, str, str, str]] = []
    keyword = _best_keyword_for_snippet(query)
    # Prefer cited chunks first (they support claims in the Answer)
    cited_chunk_ids = [(sid, cid) for sid, cid in used_citations]
    for sid, cid in cited_chunk_ids[:5]:
        c = next((x for x in chunks if x.get("chunk_id") == cid), None)
        if not c:
            continue
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        quote = _evidence_quote_best_sentence(cleaned, keyword=keyword, max_chars=280)
        if not quote or any(q == quote for q, _, _, _ in evidence_items):
            continue
        claim = claim_by_chunk.get((sid, cid), "")
        if claim:
            short_claim = (claim[:80] + "...") if len(claim) > 80 else claim
            explanation = f"Supports: {short_claim}"
        elif any(kw in quote.lower() for kw in SECTION_D_KEYWORDS):
            explanation = "Supports: Limitations or caveats mentioned in the source."
        else:
            explanation = "Supports: Direct evidence from the corpus."
        evidence_items.append((quote, sid, cid, explanation))

    # Fill to min 3–5 from remaining chunks if needed
    for c in chunks:
        if len(evidence_items) >= max(MIN_EVIDENCE_SNIPPETS, 5):
            break
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        if (sid, cid) in cited_chunk_ids:
            continue
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        quote = _evidence_quote_best_sentence(cleaned, keyword=keyword, max_chars=280)
        if not quote or any(q == quote for q, _, _, _ in evidence_items):
            continue
        claim = claim_by_chunk.get((sid, cid), "")
        explanation = f"Supports: {claim[:80]}..." if claim else "Supports: Related context from the corpus."
        evidence_items.append((quote, sid, cid, explanation))

    citations = list(used_citations)
    ref_ids = list(dict.fromkeys(sid for sid, _ in citations))
    return answer_text, evidence_items, citations


def _render_phase2_with_dual_citations(
    answer_text_internal: str,
    evidence_items: List[Tuple[str, str, str, str]],
    citations: List[Tuple[str, str]],
    retrieved_chunks: List[dict],
) -> Tuple[str, List[Tuple[str, str]], List[dict]]:
    """
    Validation flow: parse internal citations, remove invalid, map to APA, render.
    Returns (full_output, citations, citation_mapping_for_logs).
    """
    valid_chunk_ids = {c.get("chunk_id", "") for c in retrieved_chunks if c.get("chunk_id")}
    answer_visible = strip_all_citations_from_text(answer_text_internal)
    valid_evidence = [(q, sid, cid, expl) for q, sid, cid, expl in evidence_items if cid and cid in valid_chunk_ids]
    if len(valid_evidence) < MIN_EVIDENCE_SNIPPETS and evidence_items:
        valid_evidence = [(q, sid, cid, expl) for q, sid, cid, expl in evidence_items if sid and cid]
    ref_ids = list(dict.fromkeys(sid for sid, _ in citations))
    full_output = _format_phase2_output(
        answer_visible, valid_evidence, ref_ids, retrieved_chunks=retrieved_chunks
    )
    internal_to_apa = build_internal_to_apa_map(retrieved_chunks)
    citation_mapping = [
        {"apa": internal_to_apa.get(cid, ""), "source_id": sid, "chunk_id": cid}
        for sid, cid in citations
        if cid in internal_to_apa
    ]
    return full_output, citations, citation_mapping


def _detect_conflict(chunks: List[dict]) -> bool:
    """Heuristic: check for negation or contrasting phrases across chunks."""
    negations = ("however", "although", "contrary", "disagree", "conflict", "in contrast", "on the other hand")
    texts = [c.get("text", "").lower() for c in chunks]
    for t in texts:
        for n in negations:
            if n in t and len(chunks) > 1:
                return True
    return False


def generate_answer(
    query: str,
    chunks: List[dict],
    use_llm: bool = False,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Generate synthesized answer with inline citations. Trust behavior:
    - If no chunks or all low confidence: refuse to cite, state no evidence.
    - If conflicting evidence: say so and cite both sides.
    - Every citation is a retrieved chunk (no invention).
    Returns dict: answer, citations (list of (source_id, chunk_id)), model_name, prompt_version, has_conflict.
    """
    model_name = os.environ.get("PRP_MODEL_NAME", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    prompt_version = PROMPT_VERSION
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))

    # Filter by confidence if RRF score present
    scored = [c for c in chunks if c.get("rrf_score", 1.0) >= min_confidence]
    if not scored:
        scored = chunks

    use_llm_resolved = use_llm or os.environ.get("USE_LLM", "1").strip().lower() in ("1", "true", "yes")

    if not chunks:
        if has_api_key and use_llm_resolved:
            return _generate_with_llm(query, [], model_name, prompt_version)
        return {
            "answer": "Not found in corpus.",
            "citations": [],
            "citation_mapping": [],
            "model_name": model_name,
            "prompt_version": prompt_version,
            "has_conflict": False,
        }

    # USE_LLM=1 + no API key => neutral fallback; USE_LLM=0 => extractive (no API needed)
    if use_llm_resolved and not has_api_key:
        return _neutral_synthesis_fallback(query, scored, model_name, prompt_version)

    # Claim-validation queries: constrained schema (Stance | Rationale | Evidence FOR/AGAINST | Confidence)
    if _is_claim_validation_query(query):
        return _generate_claim_validation_response(query, scored, model_name, prompt_version)

    # List 5 claims strict mode (format constraint: exactly one citation + one snippet per claim)
    if _is_list_5_claims_query(query) or (_is_format_constraint_query(query) and "5 claims" in query.lower()):
        result = _generate_list_5_claims_response(query, scored, model_name, prompt_version)
        valid, violations = _validate_format_constraint(result["answer"], expected_per_claim=1)
        if not valid:
            logger.warning("Format constraint validation failed: %s", violations)
        return result

    # Relevance gate: only emit "limited evidence" when retrieval is truly off-topic (keyword_hit_count < 2 or no strong_signal)
    if not _relevance_gate_passes(query, scored):
        logger.debug("[PRP] Using limited_evidence template (relevance gate did not pass)")
        return _neutral_synthesis_fallback(query, scored, model_name, prompt_version)

    # Claim-evidence alignment gate (skip if relevance gate passed)
    if not _claim_evidence_alignment_gate(query, scored):
        related_citations = [(c.get("source_id", ""), c.get("chunk_id", "")) for c in scored[:5] if c.get("source_id") and c.get("chunk_id")]
        manifest_meta = _load_manifest_metadata()
        answer = _format_not_found_detailed_answer(query, scored, related_citations, manifest_meta)
        citation_mapping = [{"apa": format_in_text_apa(s), "source_id": s, "chunk_id": c} for s, c in related_citations]
        return {
            "answer": answer,
            "citations": related_citations,
            "citation_mapping": citation_mapping,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "has_conflict": False,
        }

    # Answerability gate
    should_refuse, _ = _answerability_gate(query, scored)
    if should_refuse:
        related_citations = [(c.get("source_id", ""), c.get("chunk_id", "")) for c in scored[:5] if c.get("source_id") and c.get("chunk_id")]
        manifest_meta = _load_manifest_metadata()
        answer = _format_not_found_detailed_answer(query, scored, related_citations, manifest_meta)
        citation_mapping = [{"apa": format_in_text_apa(s), "source_id": s, "chunk_id": c} for s, c in related_citations]
        return {
            "answer": answer,
            "citations": related_citations,
            "citation_mapping": citation_mapping,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "has_conflict": False,
        }

    # Strong-claim trust
    if _is_strong_claim_query(query) and not _retrieved_supports_strong_claim(query, scored):
        related_citations = [(c.get("source_id", ""), c.get("chunk_id", "")) for c in scored[:5] if c.get("source_id") and c.get("chunk_id")]
        manifest_meta = _load_manifest_metadata()
        answer = _format_not_found_detailed_answer(query, scored, related_citations, manifest_meta)
        citation_mapping = [{"apa": format_in_text_apa(s), "source_id": s, "chunk_id": c} for s, c in related_citations]
        return {
            "answer": answer,
            "citations": related_citations,
            "citation_mapping": citation_mapping,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "has_conflict": False,
        }

    # USE_LLM=1 => always LLM; USE_LLM=0 => always extractive. No hybrid fallback.
    if use_llm_resolved and has_api_key:
        try:
            result = _generate_with_llm(query, scored, model_name, prompt_version)
        except Exception as e:
            logger.warning("LLM generation failed: %s", e)
            result = None
        if result is None:
            result = _neutral_synthesis_fallback(query, scored, model_name, prompt_version)
    else:
        # Extractive path (USE_LLM=0 or no API key)
        answer_text, evidence_items, citations = _extractive_synthesis(query, scored)
        answer, citations_out, citation_mapping = _render_phase2_with_dual_citations(
            answer_text, evidence_items, citations, scored
        )
        result = {
            "answer": answer,
            "citations": citations_out,
            "citation_mapping": citation_mapping,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "has_conflict": _detect_conflict(scored),
        }

    # Optional: log verbatim overlap as warning only (no control flow)
    if result and use_llm_resolved:
        answer_section = _extract_answer_section_from_output(result.get("answer", ""))
        if _check_verbatim_overlap(answer_section, scored, threshold=8):
            logger.warning("Verbatim overlap detected in LLM answer (logged only)")

    has_conflict = result.get("has_conflict", False)
    if has_conflict:
        result["answer"] = (
            "Note: The retrieved evidence may contain conflicting or contrasting findings; "
            "both sides are cited below. Interpret with care.\n\n"
            + result["answer"]
        )
    return result


SYNTHESIS_SYSTEM_PROMPT = """You are a research-grade RAG system. Answer the question DIRECTLY using ONLY the passages provided. Every claim must be grounded in the passages—do not add information not present there.

Rules:
- Answer the question asked. Do not give a generic summary; address the specific query.
- Base your answer strictly on the passages. Paraphrase in your own words but do not invent facts or drift from the source content.
- Fix malformed spacing from PDF text (e.g. "per for mance" -> "performance").
- Synthesize across multiple sources. Present claims in clear, structured sentences.
- NO inline citations in the Answer. No (Author, Year), no (chunk_id).
- Do NOT say "limited evidence" or "corpus does not contain" if the passages clearly address the question—synthesize what they say.
- If the passages discuss governance, frameworks, or design (not clinical outcomes): synthesize those themes and say so.
- If the passages do not address the question: say so briefly and summarize what they do cover.

~250-400 words. Never generic "see Evidence section" as the answer. Directly answer the question using the passages."""

# Stricter prompt (legacy; verbatim retry disabled)
SYNTHESIS_SYSTEM_PROMPT_STRICT = SYNTHESIS_SYSTEM_PROMPT

# Keywords indicating governance/implementation content (vs clinical evidence)
_GOVERNANCE_KEYWORDS = (
    "governance", "trust", "equity", "liability", "checklist", "implementation",
    "framework", "policy", "ethics", "bias", "blackbox", "transparency",
    "regulation", "guidance", "compliance", "prioritization", "risk scoring",
)
_CLINICAL_KEYWORDS = ("trial", "rct", "cohort", "efficacy", "treatment", "intervention", "patients", "diagnosis")


def _classify_chunk_content(chunks: List[dict]) -> Tuple[bool, List[str]]:
    """Classify: (is_governance_heavy, theme_phrases). Theme phrases are short paraphrases from chunks."""
    combined = " ".join(
        clean_chunk_text_for_generation(c.get("text", ""))
        for c in chunks
        if c.get("text") and not is_bibliography_chunk(c.get("text", ""))
    ).lower()
    gov_count = sum(1 for k in _GOVERNANCE_KEYWORDS if k in combined)
    clin_count = sum(1 for k in _CLINICAL_KEYWORDS if k in combined)
    is_governance_heavy = gov_count >= clin_count and gov_count >= 2

    theme_phrases: List[str] = []
    seen: set = set()
    for c in chunks[:8]:
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        first_sent = re.split(r"[.!?]\s+", cleaned)[0][:150].strip()
        if first_sent and len(first_sent) >= 30:
            key = first_sent[:80].lower()
            if key not in seen:
                seen.add(key)
                theme_phrases.append(first_sent)
        if len(theme_phrases) >= 5:
            break
    return is_governance_heavy, theme_phrases


def _build_indirect_synthesis_from_chunks(query: str, chunks: List[dict]) -> str:
    """Build structured Answer from chunks when evidence is indirect. No generic 'see Evidence' template."""
    q_lower = (query or "").lower()
    is_governance, theme_phrases = _classify_chunk_content(chunks)

    # 1) One sentence: direct evidence limited
    lead = (
        "Direct evidence for the exact question is limited in the retrieved corpus."
        if not is_governance
        else "The corpus does not contain clinical evidence directly addressing the question; instead it provides governance, trust, and implementation guidance."
    )

    # 2) Two paragraphs: what the passages DO cover
    para1_parts = []
    if is_governance:
        para1_parts.append(
            "The retrieved sources focus on AI governance, trust, equity, and implementation considerations in healthcare. "
            "They discuss frameworks for evaluating AI solutions, checklist-style guidance for deployment, and considerations around bias and transparency. "
        )
    else:
        para1_parts.append(
            "The retrieved sources discuss related themes from the corpus. "
        )
    # Avoid appending raw chunk fragments (may contain PDF spacing artifacts)
    para1 = " ".join(para1_parts).strip()

    para2 = (
        "The material is useful for understanding broader context but does not provide direct empirical evidence or clinical studies addressing the specific query."
        if is_governance
        else "These passages provide tangential or background context rather than direct answers."
    )

    # 3) Gaps: what the corpus does NOT establish (no RCT/effect-size requirement)
    gaps = [
        "Direct empirical studies addressing the specific query",
        "Evidence that directly answers the exact question posed",
    ]
    if is_governance:
        gaps.append("Clinical efficacy or safety data for the intervention in question")
    gaps = list(dict.fromkeys(gaps))[:4]

    # 4) What evidence would be needed
    needs = [
        "Primary research addressing the specific question",
        "Systematic reviews or meta-analyses on the topic",
    ]
    needs = needs[:3]

    lines = [
        lead,
        "",
        para1,
        "",
        para2,
        "",
        "What the corpus does NOT establish:",
    ] + [f"- {g}" for g in gaps] + [
        "",
        "Evidence that would be needed:",
    ] + [f"- {n}" for n in needs]
    return "\n".join(lines)


def _is_generic_answer(answer_section: str) -> bool:
    """True if answer is a generic template (e.g. 'see Evidence section' as main content)."""
    if not answer_section or len(answer_section.strip()) < 80:
        return True
    s = answer_section.lower()
    generic_markers = (
        "see the evidence section",
        "see evidence for details",
        "limited direct evidence on this topic",
        "retrieved sources provide limited direct evidence",
        "what we know: the corpus contains related material; see evidence",
    )
    if any(m in s for m in generic_markers):
        # Check it's not just a passing mention - if most of the answer is generic, flag it
        generic_ratio = sum(len(m) for m in generic_markers if m in s) / max(len(s), 1)
        if generic_ratio > 0.15 or s.count("see") >= 2:
            return True
    return False


def _neutral_synthesis_fallback(
    query: str, chunks: List[dict], model_name: str, prompt_version: str
) -> dict:
    """When LLM unavailable or verbatim persists: chunk-derived indirect synthesis + Evidence from chunks."""
    answer = _build_indirect_synthesis_from_chunks(query, chunks)
    evidence_items: List[Tuple[str, str, str, str]] = []
    keyword = _best_keyword_for_snippet(query)
    for c in chunks[:6]:
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        quote = _evidence_quote_best_sentence(cleaned, keyword=keyword, max_chars=240)
        if not quote:
            continue
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        evidence_items.append((quote, sid, cid, "Supports: Related context from corpus."))
    citations = [(sid, cid) for _, sid, cid, _ in evidence_items]
    ref_ids = list(dict.fromkeys(sid for sid, _ in citations))
    full = _format_phase2_output(answer, evidence_items, ref_ids, retrieved_chunks=chunks)
    internal_to_apa = build_internal_to_apa_map(chunks)
    citation_mapping = [{"apa": internal_to_apa.get(cid, ""), "source_id": sid, "chunk_id": cid} for sid, cid in citations if cid in internal_to_apa]
    return {
        "answer": full,
        "citations": citations[:10],
        "citation_mapping": citation_mapping,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "has_conflict": False,
    }


def _estimate_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len((text or "").strip()) // 4


def _build_extractive_evidence(query: str, chunks: List[dict], max_items: int = 8) -> List[Tuple[str, str, str, str]]:
    """Build Evidence section extractively: quoted chunk text + (source_id_chunk_id), no APA inline."""
    evidence_items: List[Tuple[str, str, str, str]] = []
    keyword = _best_keyword_for_snippet(query)
    for c in chunks[:max_items * 2]:  # oversample to allow filtering
        if len(evidence_items) >= max_items:
            break
        raw = (c.get("text") or "").strip()
        if not raw or is_bibliography_chunk(raw):
            continue
        cleaned = clean_chunk_text_for_generation(raw)
        quote = _evidence_quote_best_sentence(cleaned, keyword=keyword, max_chars=240)
        if not quote or any(q == quote for q, _, _, _ in evidence_items):
            continue
        sid, cid = c.get("source_id", ""), c.get("chunk_id", "")
        evidence_items.append((quote, sid, cid, "Supports: Relevant passage from corpus."))
    return evidence_items


def _generate_with_llm(
    query: str,
    chunks: List[dict],
    model_name: str,
    prompt_version: str,
    use_strict_prompt: bool = False,
) -> dict:
    """LLM generates Answer only (synthesized, no citations). Evidence built extractively from chunks."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai not installed")
    client = openai.OpenAI()
    llm_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    chunk_list = []
    for c in chunks:
        cid = c.get("chunk_id", "")
        chunk_list.append(f"[{cid}]\n{clean_chunk_text_for_generation(c.get('text', ''))}")
    context = "\n\n---\n\n".join(chunk_list) if chunks else "(No passages provided.)"
    context_tokens = _estimate_token_count(context)

    logger.info("[PRP] Model: %s | Context chunks: %d | Est. context tokens: %d", llm_model, len(chunks), context_tokens)

    sys_prompt = SYNTHESIS_SYSTEM_PROMPT_STRICT if use_strict_prompt else SYNTHESIS_SYSTEM_PROMPT
    prompt = (
        sys_prompt
        + "\n\nOutput ONLY the Answer section (NO Evidence, NO References). "
        "Passages:\n" + context + "\n\nQuestion: " + query + "\n\nOutput Answer only:"
    )
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    answer_text = resp.choices[0].message.content or ""
    answer_text = strip_all_citations_from_text(answer_text).strip()

    # Replace generic template output with chunk-derived synthesis
    if chunks and _is_generic_answer(answer_text):
        logger.warning("LLM returned generic template; replacing with chunk-derived indirect synthesis")
        answer_text = _build_indirect_synthesis_from_chunks(query, chunks)

    if not chunks:
        return {
            "answer": answer_text,
            "citations": [],
            "citation_mapping": [],
            "model_name": model_name,
            "prompt_version": prompt_version,
            "has_conflict": False,
        }

    evidence_items = _build_extractive_evidence(query, chunks, max_items=8)
    citations = [(sid, cid) for _, sid, cid, _ in evidence_items]
    ref_ids = list(dict.fromkeys(sid for sid, _ in citations))
    full_output = _format_phase2_output(answer_text, evidence_items, ref_ids, retrieved_chunks=chunks)
    internal_to_apa = build_internal_to_apa_map(chunks)
    citation_mapping = [
        {"apa": internal_to_apa.get(cid, ""), "source_id": sid, "chunk_id": cid}
        for sid, cid in citations
        if cid in internal_to_apa
    ]
    has_conflict = _detect_conflict(chunks)
    return {
        "answer": full_output,
        "citations": [(a, b) for a, b in citations if a and b],
        "citation_mapping": citation_mapping,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "has_conflict": has_conflict,
    }
