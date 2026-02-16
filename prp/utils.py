"""Utilities: source_id, logging."""
import hashlib
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timezone

from .config import RUNS_DIR, PHASE2_RUNS_JSONL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prp")


def source_id_from_path(file_path: Path) -> str:
    """Deterministic source_id from filename: sanitize + short hash."""
    name = file_path.name
    # Sanitize: keep alphanumeric, hyphen, underscore
    base = re.sub(r"[^\w\-.]", "_", name)
    base = base[:80]
    raw = f"{base}_{file_path.stat().st_size}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{base[:30]}_{h}".strip("_")


def append_run_log(record: dict) -> None:
    """Append a single JSON object as one line to phase2_runs.jsonl."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE2_RUNS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_run_to_file(record: dict, metrics: dict | None = None) -> Path:
    """
    Save query run to logs/runs/<timestamp>.json per Phase 2 spec.
    record: query_id, query_text, retrieved_chunks, model_version, prompt_version, answer_text, citations_used
    metrics: optional per-query metrics for eval runs
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    import time
    ts_ms = f"{ts}.{int(time.time() * 1000) % 1000:03d}"
    out = dict(record)
    out["model_version"] = out.get("model_name") or out.get("model_version", "")
    out["answer_text"] = out.get("generated_answer") or out.get("answer_text", "")
    out["citations_used"] = out.get("citations") or out.get("citations_used", [])
    if isinstance(out.get("citations_used"), list) and out["citations_used"] and isinstance(out["citations_used"][0], (list, tuple)):
        out["citations_used"] = [list(c) for c in out["citations_used"]]
    if metrics:
        out["metrics_if_eval_query"] = metrics
    path = RUNS_DIR / f"{ts_ms}.json"
    # Handle rare collision
    idx = 0
    while path.exists():
        path = RUNS_DIR / f"{ts_ms}_{idx}.json"
        idx += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_question(q: str) -> str:
    """Strip trailing punctuation and normalize whitespace before inserting query into output."""
    if not q or not isinstance(q, str):
        return ""
    s = q.strip()
    s = re.sub(r"\s+", " ", s)
    while s and s[-1] in ".,;:?!":
        s = s[:-1].rstrip()
    return s


def fix_concatenated_words(text: str) -> str:
    """
    Fix common PDF extraction artifacts where spaces are missing between words.
    E.g. 'AItools' -> 'AI tools', 'patientcentered' -> 'patient-centered', 'thatcan' -> 'that can'.
    """
    if not text:
        return text
    # AI/ML + word: AItools, XAItools, MLenabled, etc.
    text = re.sub(r"\b(AI|XAI|ML)([a-z][a-z]+)\b", r"\1 \2", text, flags=re.IGNORECASE)
    # Direct replacements for common concatenations (order matters for overlaps)
    replacements = [
        ("AItools", "AI tools"),
        ("XAIdsystems", "XAI systems"),
        ("includingbias", "including bias"),
        ("mimichuman", "mimic human"),
        ("withouthumanoversight", "without human oversight"),
        ("includingthefirst", "including the first"),
        ("thelevel", "the level"),
        ("thelevels", "the levels"),
        ("morethan", "more than"),
        ("thatcan", "that can"),
        ("forunderserved", "for underserved"),
        ("patientcentered", "patient-centered"),
        ("machinelearning", "machine learning"),
        ("tomitigate", "to mitigate"),
        ("etal.", "et al."),
        ("coreideas", "core ideas"),
        ("identifiedafter", "identified after"),
        ("questionwas", "question was"),
        ("outlinedsteps", "outlined steps"),
        ("healthcan", "health can"),
        ("onesizefitsall", "one size fits all"),
        ("havebeen", "have been"),
        ("canbe", "can be"),
        ("theybeen", "they been"),
        ("impactof", "impact of"),
        ("ai ms", "aims"),
        ("In for matics", "Informatics"),
        ("Trans for mer", "Transformer"),
        ("monitoringper", "monitoring per"),
    ]
    for bad, good in replacements:
        text = text.replace(bad, good)
    return text


def normalize_extracted_text(text: str) -> str:
    """
    Normalize PDF-extracted text before chunking. Applied after extraction (e.g. in ingest)
    so sources.jsonl and chunks get clean text.
    - Fix concatenated words (AItools -> AI tools, etc.).
    - Remove soft hyphen (U+00AD).
    - Join hyphenated line breaks: (\\w)-\\n(\\w) -> \\1\\2.
    - Replace newlines inside paragraphs with spaces (keep paragraph breaks).
    - Collapse repeated whitespace.
    """
    if not text or not isinstance(text, str):
        return text
    # 0) Fix missing spaces between words (PDF extraction artifact)
    text = fix_concatenated_words(text)
    # 1) Remove soft hyphens (U+00AD)
    text = text.replace("\u00ad", "")
    # 2) Join hyphenated line breaks: word-\nnext -> wordnext (only when \w before and after)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # 3) Merge lines when previous ends mid-word and current starts with lowercase fragment
    lines = text.split("\n")
    result = []
    for line in lines:
        if not result:
            result.append(line)
            continue
        prev = result[-1]
        if prev and line and prev[-1:].isalpha() and line[0:1].islower():
            result[-1] = prev + line
        else:
            result.append(line)
    # 4) Replace newlines inside paragraphs with spaces (keep paragraph breaks)
    merged = "\n".join(result)
    paras = re.split(r"\n{2,}", merged)
    paras = [re.sub(r"\n", " ", p) for p in paras]
    normalized = "\n\n".join(paras)
    # 5) Collapse repeated whitespace (preserve paragraph breaks)
    paras = normalized.split("\n\n")
    paras = [re.sub(r"\s+", " ", p).strip() for p in paras]
    return "\n\n".join(paras).strip()


# Boilerplate line patterns (drop lines that match any of these)
_BOILERPLATE_PATTERNS = [
    r"page\s*number\s*not\s*for\s*citation\s*purposes",
    r"JOURNAL\s*OF\s*MEDICAL\s*INTERNET\s*RESEARCH",
    r"J\s*Med\s*Internet\s*Res",
    r"XSL[•·]\s*FO",  # XSL•FO or XSL·FO (bullet/middle dot)
    r"RenderX",
    r"doi\.org",
    r"doi\s*:",
    r"https?://",
    r"Volume\s*\d",  # Volume 30 | July 2024 | 1888–1897
    r"\d+\s*\|\s*[A-Za-z]+\s*\d{4}\s*\|\s*\d+[–\-]\d+",  # header fragment with page range
    r"^\s*\d{4}\s*[–\-]\s*\d{4}\s*$",  # page range 1888–1897
    r"^\s*\d+\s*[–\-]\s*\d+\s*$",  # page range like 1-10
    r"nature\s*medicine",
    r"\bvol\.\s*\d",
    r"^\s*Article\s*$",
    r"^\s*\d+\s*\|\s*\w+",
    r"^\s*\[?\d+\]?\s*$",  # standalone number or [1]
]
# Figure/table caption lines: drop lines containing these
_CAPTION_LINE_PATTERNS = [
    r"\bFigure\s+\d",
    r"\bTable\s+\d",
    r"Example\s+of\s+composite\s+score",
    r"HDL\s*:",
]
_BOILERPLATE_RE = re.compile("|".join(f"({p})" for p in _BOILERPLATE_PATTERNS), re.IGNORECASE)
_CAPTION_LINE_RE = re.compile("|".join(f"({p})" for p in _CAPTION_LINE_PATTERNS), re.IGNORECASE)


def _is_boilerplate_line(line: str) -> bool:
    """True if line looks like PDF boilerplate or figure/table caption."""
    t = line.strip()
    if not t or len(t) < 3:
        return False
    if _BOILERPLATE_RE.search(t):
        return True
    if _CAPTION_LINE_RE.search(t):
        return True
    return False


# Pre-chunk filter: drop lines likely to be acknowledgements/funding/conflicts (before chunking).
# Rationale: these lines harm groundedness and answer usefulness when retrieved and cited.
# Kept conservative so we do not remove legitimate methods/results (e.g. "acknowledged the limitations").
# Exposed for check scripts (e.g. scripts/check_chunks_ack_filter.py).
ACK_FUNDING_PATTERNS = [
    "supported by",
    "funded by",
    "conflict of interest",
    "conflicts of interest",
    "TKI ",  # Dutch funding body (space to avoid mid-word)
    "Topsector",
    "grant number",
    "grant no.",
    "grant from",
    "grant by",
    "grant #",
    "Received:",
    "Accepted:",
    "Published:",  # standalone metadata line
    "frontiersin.org",
    "mdpi.com",
    "springer.com",
]
# "acknowledg" only when line looks like ack section (short or contains thank/support/fund)
_ACK_ALWAYS_RE = re.compile(
    "|".join(re.escape(p) for p in ACK_FUNDING_PATTERNS),
    re.IGNORECASE,
)


def _is_ack_funding_line(line: str) -> bool:
    """True if line should be dropped as acknowledgement/funding/boilerplate."""
    t = line.strip()
    if not t:
        return False
    if _ACK_ALWAYS_RE.search(t):
        return True
    # "acknowledg" only drop when line is short or has thank/support/fund (avoid "acknowledged the limitations")
    if "acknowledg" in t.lower():
        if len(t) <= 120:
            return True
        if any(x in t.lower() for x in ("thank", "supported by", "funded by", "contribution")):
            return True
    return False


# Chunk-level boilerplate: drop chunks that are predominantly ack/funding, methodology, reference list, or figure/table-only.
_IS_BOILERPLATE_ACK = re.compile(
    r"supported\s+by|funded\s+by|\bgrant\b|conflict(s)?\s+of\s+interest",
    re.IGNORECASE,
)
_IS_BOILERPLATE_METHODOLOGY = re.compile(
    r"Stage\s+[1-5](?:\s*[:/])?|collating,?\s+summarizing|PRISMA|\bwe\s+searched\b|inclusion\s+criteria",
    re.IGNORECASE,
)
# Reference list: numbered refs (1. Author, 2. Author) or many semicolons or Received:
_IS_BOILERPLATE_REF_AUTHOR_YEAR = re.compile(r"\b\d+\.\s+[A-Z][a-z]+")
_IS_BOILERPLATE_RECEIVED = re.compile(r"Received\s*:", re.IGNORECASE)
# Figure/table-only: chunk is mostly Figure N, Table N, Appendix
_IS_BOILERPLATE_FIGURE_TABLE = re.compile(r"\b(Figure|Table|Appendix)\s*\d*\b", re.IGNORECASE)


# Bibliography/reference section detection: headings and citation-like patterns
BIBLIOGRAPHY_HEADINGS = (
    "references",
    "bibliography",
    "works cited",
    "reference list",
    "references and notes",
)
_BIB_HEADING_RE = re.compile(
    r"^(?:\s*\d*\.?\s*)?(?:references|bibliography|works\s+cited|reference\s+list)\s*:?\s*$",
    re.IGNORECASE,
)
_BIB_PATTERNS = [
    r"Press,?\s+Washington,?\s+DC",
    r"National\s+Academies",
    r"\bISBN\b",
    r"doi\s*:",
    r"doi\.org",
    r"Retrieved\s+from",
    r"Available\s+at",
    r"et\s+al\.\s*[\[\(]?\d{4}",
    r"\(\d{4}\)",
    r"\d{4}\s*;\s*\d+",
    r"vol\.\s*\d+\s*[\[\(]?\d+[\]\)]?",
    r"pp\.\s*\d+",
    r"pp\s*\d+",
    r"\b\d+\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+",
    r"^\[\d+\]\s+",
    r"^\s*\d+\.\s+[A-Z]",
]
_BIB_PATTERNS_RE = re.compile("|".join(f"({p})" for p in _BIB_PATTERNS), re.IGNORECASE | re.MULTILINE)


def bibliography_score(text: str) -> float:
    """
    Return a score 0.0–1.0 indicating how likely the chunk is from References/Bibliography.
    Uses heuristics: section headings, citation patterns, semicolon density, author-year patterns.
    Chunks with score >= threshold should be filtered from retrieval.
    """
    if not text or not isinstance(text, str):
        return 0.0
    t = text.strip()
    if len(t) < 20:
        return 0.0
    score = 0.0
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # Heading check: "References", "Bibliography", etc.
    for ln in lines[:5]:  # first few lines
        if _BIB_HEADING_RE.match(ln):
            return 1.0
        if any(h in ln.lower() for h in BIBLIOGRAPHY_HEADINGS) and len(ln) < 80:
            score += 0.4

    # Pattern matches (citation-like fragments)
    pattern_matches = len(_BIB_PATTERNS_RE.findall(t))
    if pattern_matches >= 3:
        score += 0.5
    elif pattern_matches >= 1:
        score += 0.2

    # Semicolon density (citation lists use many semicolons)
    if t.count(";") >= 5:
        score += 0.25
    elif t.count(";") >= 3:
        score += 0.1

    # Author-year patterns: (2020), (Smith, 2020), 2020; 24
    author_year = len(re.findall(r"\(\s*\d{4}\s*\)", t)) + len(re.findall(r"\d{4}\s*;\s*\d+", t))
    if author_year >= 3:
        score += 0.35
    elif author_year >= 1:
        score += 0.15

    # Lines starting with "1.", "2.", "[12]"
    numbered_ref_lines = sum(
        1 for ln in lines if re.match(r"^(\s*\d+\.\s+|\s*\[\d+\]\s+)", ln)
    )
    if numbered_ref_lines >= 2:
        score += 0.4
    elif numbered_ref_lines >= 1 and len(lines) <= 4:
        score += 0.3

    return min(1.0, score)


def is_bibliography_chunk(text: str, threshold: float = 0.5) -> bool:
    """
    True if chunk is likely from References/Bibliography section.
    Uses bibliography_score; chunks with score >= threshold are filtered.
    Default threshold 0.5 balances precision/recall for typical reference list fragments.
    """
    return bibliography_score(text) >= threshold


def is_boilerplate(text: str) -> bool:
    """
    True if chunk text is predominantly boilerplate and should be dropped before retrieval ranking
    or final selection. Matches: acknowledgements/funding, scoping review methodology boilerplate,
    reference list fragments, figure/table-only lines, bibliography chunks.
    """
    if not text or not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) < 10:
        return True

    # Acknowledgements/funding
    if _IS_BOILERPLATE_ACK.search(t):
        return True

    # Scoping review methodology (Stage 1/2/3/4/5, collating summarizing, PRISMA, we searched, inclusion criteria)
    if _IS_BOILERPLATE_METHODOLOGY.search(t):
        return True

    # Reference list fragments: multiple author-year tokens, many semicolons, or Received:
    author_year_matches = _IS_BOILERPLATE_REF_AUTHOR_YEAR.findall(t)
    if len(author_year_matches) >= 3:
        return True
    if t.count(";") >= 5:
        return True
    if _IS_BOILERPLATE_RECEIVED.search(t):
        return True

    # Figure/table-only: chunk is short and is mainly Figure/Table/Appendix
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    figure_table_lines = sum(1 for ln in lines if _IS_BOILERPLATE_FIGURE_TABLE.search(ln))
    if figure_table_lines >= 2:
        return True
    if figure_table_lines >= 1 and len(t) < 350:
        return True

    # Bibliography/reference section chunks (similar to ACK/FUNDING filter)
    if is_bibliography_chunk(text, threshold=0.5):
        return True

    return False


def is_boilerplate_with_bib_threshold(text: str, bibliography_threshold: float = 0.5) -> bool:
    """
    Same as is_boilerplate but with configurable bibliography threshold.
    Used in retrieval fallback: if too few chunks pass, retry with higher threshold (e.g. 0.7).
    """
    if not text or not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) < 10:
        return True
    if _IS_BOILERPLATE_ACK.search(t):
        return True
    if _IS_BOILERPLATE_METHODOLOGY.search(t):
        return True
    author_year_matches = _IS_BOILERPLATE_REF_AUTHOR_YEAR.findall(t)
    if len(author_year_matches) >= 3:
        return True
    if t.count(";") >= 5:
        return True
    if _IS_BOILERPLATE_RECEIVED.search(t):
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    figure_table_lines = sum(1 for ln in lines if _IS_BOILERPLATE_FIGURE_TABLE.search(ln))
    if figure_table_lines >= 2:
        return True
    if figure_table_lines >= 1 and len(t) < 350:
        return True
    if is_bibliography_chunk(text, threshold=bibliography_threshold):
        return True
    return False


# Evidence strength per chunk: RCT > Review > Observational/Other
EVIDENCE_STRENGTH_RCT = "RCT"
EVIDENCE_STRENGTH_REVIEW = "Review"
EVIDENCE_STRENGTH_OTHER = "Observational/Other"

# Evidence type labels for "Evidence mix" output
EVIDENCE_TYPE_RCT = "RCT"
EVIDENCE_TYPE_OBSERVATIONAL = "observational"
EVIDENCE_TYPE_SYSTEMATIC_REVIEW = "systematic review"
EVIDENCE_TYPE_SCOPING_REVIEW = "scoping review"
EVIDENCE_TYPE_EDITORIAL_OTHER = "editorial/other"


def evidence_type_label(text: str, title: str = "") -> str:
    """
    Classify chunk/source by study type for Evidence mix counts.
    Returns one of: RCT | observational | systematic review | scoping review | editorial/other
    Uses simple keyword rules from title + chunk text.
    Source of truth: chunk text first; manifest type/title can override when passed via title.
    """
    combined = ((title or "") + " " + (text or "")).lower()
    if not combined.strip():
        return EVIDENCE_TYPE_EDITORIAL_OTHER
    # Check in order of specificity (more specific first)
    if "randomized controlled trial" in combined or "randomised controlled trial" in combined:
        return EVIDENCE_TYPE_RCT
    if "randomized" in combined or "randomised" in combined:
        if "trial" in combined or "controlled" in combined:
            return EVIDENCE_TYPE_RCT
    if "systematic review" in combined or "systematic literature" in combined:
        return EVIDENCE_TYPE_SYSTEMATIC_REVIEW
    if "scoping review" in combined or "scoping literature" in combined:
        return EVIDENCE_TYPE_SCOPING_REVIEW
    if "meta-analysis" in combined or "meta analysis" in combined:
        return EVIDENCE_TYPE_SYSTEMATIC_REVIEW
    if "observational" in combined or "cohort" in combined or "cross-sectional" in combined:
        return EVIDENCE_TYPE_OBSERVATIONAL
    if "trial" in combined and ("phase" in combined or "clinical trial" in combined):
        return EVIDENCE_TYPE_RCT
    # Standalone "RCT" (e.g. "this RCT", "RCTs demonstrated")
    if re.search(r"\brct\b", combined):
        return EVIDENCE_TYPE_RCT
    return EVIDENCE_TYPE_EDITORIAL_OTHER


def evidence_strength(text: str) -> str:
    """
    Tag chunk by evidence type using simple heuristics:
    - "randomized controlled trial" or "trial" → RCT
    - "systematic review", "meta-analysis", "scoping review" → Review
    - else → Observational/Other
    """
    if not text or not isinstance(text, str):
        return EVIDENCE_STRENGTH_OTHER
    t = text.lower()
    if "randomized controlled trial" in t or "randomised controlled trial" in t:
        return EVIDENCE_STRENGTH_RCT
    if " trial " in t or " trials " in t or t.strip().endswith(" trial") or t.startswith("trial "):
        return EVIDENCE_STRENGTH_RCT
    if "systematic review" in t or "meta-analysis" in t or "meta analysis" in t or "scoping review" in t:
        return EVIDENCE_STRENGTH_REVIEW
    return EVIDENCE_STRENGTH_OTHER


def filter_acknowledgements_and_boilerplate(text: str) -> str:
    """
    Remove lines likely to be acknowledgements, funding, conflicts of interest,
    or publisher/metadata boilerplate. Applied BEFORE chunking so such text
    never enters chunks.jsonl. Conservative: only drops lines matching known
    patterns; does not remove legitimate methods/results.
    """
    if not text or not isinstance(text, str):
        return text
    lines = text.split("\n")
    kept = [ln for ln in lines if not _is_ack_funding_line(ln)]
    return "\n".join(kept)


# Phrases to strip from text (PDF renderer boilerplate that may appear mid-line)
_BOILERPLATE_PHRASES = [
    r"page\s*number\s*not\s*for\s*citation\s*purposes",
    r"JOURNAL\s*OF\s*MEDICAL\s*INTERNET\s*RESEARCH",
    r"J\s*Med\s*Internet\s*Res\s*\d{4}\s*\|\s*vol\.\s*\d+",
    r"XSL[•·]\s*FO",
    r"RenderX",
    r"Volume\s*\d+\s*\|\s*[A-Za-z]+\s*\d{4}\s*\|\s*\d+[–\-]\d+",  # Volume 30 | July 2024 | 1888–1897
]
_BOILERPLATE_PHRASES_RE = re.compile("|".join(f"({p})" for p in _BOILERPLATE_PHRASES), re.IGNORECASE)

# Short words we must not join with the following token (avoid "in order" -> "inorder", "Publications that" -> "Publicationsthat")
_CLEAN_TEXT_NO_JOIN_FIRST = frozenset(
    "in on at to for of the and or is it as an be by if no so up we he we my me".split()
)
_CLEAN_TEXT_NO_JOIN_SECOND = frozenset(
    "the and that may not with for to of in on at is it as or were plans data any areas".split()
)

# Section-header phrases to strip when at sentence start and followed by another capitalized phrase
_LEADING_HEADER_PHRASES = (
    "Personalized Nutrition", "Introduction", "Background", "Methods", "Results",
    "Discussion", "Conclusion", "Abstract", "Objective", "Evidence", "Summary",
)
# Only strip when phrase is followed by punctuation or a capital (new sentence), not "may vary" etc.
# (?-i:[A-Z]) = case-sensitive lookahead so we don't strip "personalized nutrition may"
_LEADING_HEADER_RE = re.compile(
    r"(^|[.!?]\s+)(?:" + "|".join(re.escape(p) for p in _LEADING_HEADER_PHRASES) + r")\s*[,.]?\s+(?=(?-i:[A-Z]))",
    re.IGNORECASE,
)

# Placeholder to preserve paragraph breaks when collapsing whitespace (must not appear in normal text)
_PARA_PLACEHOLDER = "\x00\x01\x00"

# Known PDF-merge clumps (first, second) to split: "firstsecond" -> "first second" (conservative list)
_CLEAN_TEXT_UNMERGE_PAIRS = [
    ("personalized", "meal"), ("Publications", "that"), ("not", "very"),
    ("group", "level"), ("food", "group"), ("nutrition", "systems"),
    ("responses", "to"), ("Patient", "data"), ("mention", "any"), ("disease", "areas"),
]

# Stopwords that may be embedded when line-break boundary was lost (split only when safe)
_CLEAN_TEXT_EMBEDDED_STOPWORDS = ("the", "and", "that", "may", "not", "with", "for")
# Don't split when these would be created: "other" (o+the+r), "grand" (gr+and), "within" (wi+th+in)
_CLEAN_TEXT_NO_SPLIT_BEFORE = {"the": ("o",), "and": ("gr", "br", "st"), "with": ("wi",)}
_CLEAN_TEXT_NO_SPLIT_AFTER = {"the": ("r",), "and": (), "that": (), "may": (), "not": (), "with": ("in",), "for": ()}


def _split_embedded_stopword(m: re.Match) -> str:
    """Split clumped token like 'nutritionmay' -> 'nutrition may'; avoid splitting 'other', 'grand', 'within'."""
    before, sw, after = m.group(1), m.group(2), m.group(3)
    sw_lower = sw.lower()
    if len(before) < 2:
        return m.group(0)
    if len(after) > 0 and len(after) < 2:
        return m.group(0)
    if sw_lower in _CLEAN_TEXT_NO_SPLIT_BEFORE and any(before.lower().endswith(x) for x in _CLEAN_TEXT_NO_SPLIT_BEFORE[sw_lower]):
        return m.group(0)
    if sw_lower in _CLEAN_TEXT_NO_SPLIT_AFTER and after and any(after.lower().startswith(x) for x in _CLEAN_TEXT_NO_SPLIT_AFTER[sw_lower]):
        return m.group(0)
    return before + " " + sw + " " + after


def _join_broken_word(m: re.Match) -> str:
    """Join two tokens only if they look like a single word split by whitespace (e.g. clini cal -> clinical).
    Do not join when either part is a common short word, or when second part is long (likely a real word)."""
    first, second = m.group(1), m.group(2)
    if first.lower() in _CLEAN_TEXT_NO_JOIN_FIRST:
        return m.group(0)
    if second.lower() in _CLEAN_TEXT_NO_JOIN_FIRST or second.lower() in _CLEAN_TEXT_NO_JOIN_SECOND:
        return m.group(0)
    if len(first) + len(second) > 20:
        return m.group(0)
    # Only join when second part is short (e.g. "cal", "ness") to avoid "maintain health" -> "maintainhealth"
    if len(second) > 5:
        return m.group(0)
    return first + second


# PDF artifacts: wrongly split words (extra spaces) -> correct
_CLEAN_CHUNK_JOIN_WORDS = [
    ("per for mance", "performance"), ("per for med", "performed"),
]
# Additional PDF merge pairs for generation (applied after clean_text in clean_chunk_text_for_generation)
_CLEAN_CHUNK_EXTRA_UNMERGE = [
    ("apparent", "from"), ("such", "as"), ("for", "example"), ("in", "order"),
    ("as", "such"), ("as", "well"), ("that", "is"), ("for", "instance"),
    ("that", "such"), ("due", "to"), ("based", "on"), ("rather", "than"),
    ("such", "that"), ("in", "addition"), ("as", "a"), ("part", "of"),
    ("prior", "to"), ("may", "lead"), ("in", "consistencies"),
    ("across", "both"),
]


def clean_chunk_text_for_generation(s: str) -> str:
    """
    Deterministic cleaning of retrieved chunk text before sending to LLM.
    Fixes PDF artifacts: missing spaces, hyphenation, line breaks, whitespace.
    Use this (not clean_text) when chunks are passed to generation.
    """
    if not s or not isinstance(s, str):
        return ""
    out = fix_concatenated_words(s)
    out = clean_text(out)
    # Fix wrongly split words (e.g. "per for mance" -> "performance")
    for wrong, correct in _CLEAN_CHUNK_JOIN_WORDS:
        out = out.replace(wrong, correct)
    # Extra unmerge for common PDF concatenations
    for a, b in _CLEAN_CHUNK_EXTRA_UNMERGE:
        out = re.sub(re.escape(a) + re.escape(b), a + " " + b, out, flags=re.IGNORECASE)
    # Collapse any double spaces from unmerge
    out = re.sub(r"\s+", " ", out).strip()
    return out


def clean_text(s: str) -> str:
    """
    Clean chunk text: remove boilerplate, bracket citations, hyphenated line breaks,
    normalize newlines (paragraph-aware), fix missing spaces, join broken words.
    Does NOT break abbreviations like U.S. or normal camelCase.
    """
    if not s or not isinstance(s, str):
        return ""
    # Drop boilerplate and caption lines
    lines = [ln for ln in s.splitlines() if not _is_boilerplate_line(ln)]
    s = "\n".join(lines)
    # Remove boilerplate phrases that may appear mid-line
    s = _BOILERPLATE_PHRASES_RE.sub("", s)
    # Remove paper-internal bracket citations: [66], [66,67], etc.
    s = re.sub(r"\[\d+(?:[,\s\-]*\d+)*\]", "", s)
    # Remove soft hyphens (U+00AD)
    s = s.replace("\u00ad", "")
    # 1) Join hyphenated line breaks: "word-\nnext" -> "wordnext" (keep as one word)
    s = re.sub(r"-\s*\n\s*", "", s)
    # 2) Paragraph-aware newlines: normalize 2+ newlines to paragraph break, then single \n -> space
    s = re.sub(r"\n{2,}", _PARA_PLACEHOLDER, s)
    s = s.replace("\n", " ")
    s = s.replace(_PARA_PLACEHOLDER, "\n\n")
    # 3) Collapse multiple spaces (but preserve \n\n)
    s = s.replace("\n\n", _PARA_PLACEHOLDER)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(_PARA_PLACEHOLDER, "\n\n")
    # 4) Conservative unmerge of known PDF clumps
    for a, b in _CLEAN_TEXT_UNMERGE_PAIRS:
        s = re.sub(re.escape(a) + re.escape(b), a + " " + b, s, flags=re.IGNORECASE)
    # 5) Space after punctuation when followed by letter (do NOT touch period -> preserve U.S.)
    s = re.sub(r"(\w)(\()", r"\1 \2", s)
    s = re.sub(r"(\w)(\[)", r"\1 \2", s)
    s = re.sub(r"(\w)(\{)", r"\1 \2", s)
    s = re.sub(r"(\))(\w)", r"\1 \2", s)
    s = re.sub(r"(\])(\w)", r"\1 \2", s)
    s = re.sub(r"(\})(\w)", r"\1 \2", s)
    s = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", s)
    # 6) Space between lowercase and uppercase when stuck: "NutritionAdvice" -> "Nutrition Advice"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # 7) Split embedded stopwords: "...tionmay..." -> "...tion may..." (avoid "other", "grand", "within")
    _embedded_re = re.compile(
        r"(\w{2,})(" + "|".join(re.escape(w) for w in _CLEAN_TEXT_EMBEDDED_STOPWORDS) + r")(\w*)",
        re.IGNORECASE,
    )
    s = _embedded_re.sub(_split_embedded_stopword, s)
    # 8) Join broken words only when safe: "clini cal" -> "clinical"
    s = re.sub(r"\b([a-zA-Z]{2,12}) ([a-z][a-zA-Z]{1,12})\b", _join_broken_word, s)
    # 9) Collapse spaces again (preserve paragraph breaks)
    s = s.replace("\n\n", _PARA_PLACEHOLDER)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(_PARA_PLACEHOLDER, "\n\n")
    # Remove leading section headers at sentence boundaries
    s = _LEADING_HEADER_RE.sub(r"\1", s)
    return s.strip()


def preview_from_clean(clean_s: str, max_chars: int = 200) -> str:
    """Return a preview of at most max_chars, ending at a word boundary (no mid-word cut)."""
    if not clean_s or max_chars <= 0:
        return ""
    if len(clean_s) <= max_chars:
        return clean_s
    segment = clean_s[: max_chars + 1]
    last_space = segment.rfind(" ")
    if last_space > max_chars // 2:
        return segment[:last_space].strip()
    return segment[:max_chars].strip()


def evidence_snippet(clean_s: str, min_chars: int = 120, max_chars: int = 240) -> str:
    """
    Return a readable evidence quote: do NOT start from char 0. Start from first sentence
    boundary, or first capital after index 0, or first space after 20 chars. Snippet is
    min_chars–max_chars and ends at a sentence boundary if possible.
    """
    if not clean_s or max_chars <= 0:
        return ""
    start = 0
    if len(clean_s) > 30:
        # Prefer: start after first sentence boundary (e.g. ". " or "? ")
        m = re.search(r"[.!?]\s+([A-Z])", clean_s)
        if m:
            start = m.start(1)
        else:
            # First capital letter after index 0
            m2 = re.search(r"[A-Z]", clean_s[1:])
            if m2:
                start = 1 + m2.start()
            elif len(clean_s) > 25:
                # First whitespace after 20 chars
                idx = clean_s.find(" ", 20)
                if idx != -1:
                    start = idx + 1
    segment = clean_s[start : start + max_chars + 1]
    # End at sentence boundary if possible
    last_sent = max(
        segment.rfind(". "),
        segment.rfind("! "),
        segment.rfind("? "),
    )
    if last_sent >= min_chars:
        segment = segment[: last_sent + 1].strip()
    else:
        # End at word boundary
        last_space = segment.rfind(" ")
        if last_space >= min_chars:
            segment = segment[:last_space].strip()
        elif len(segment) > max_chars:
            segment = segment[:max_chars].strip()
    if len(segment) < min_chars and len(clean_s) > start + min_chars:
        segment = clean_s[start : start + max_chars]
        last_space = segment.rfind(" ")
        if last_space >= min_chars:
            segment = segment[:last_space].strip()
    return segment.strip()
