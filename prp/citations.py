"""
APA citation formatting. Loads source metadata from manifest.csv and sources.jsonl.
Provides format_in_text_apa() and format_reference_apa() for structured output.
"""
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import MANIFEST_PATH, SOURCES_JSONL

# source_id -> metadata dict (authors, year, title, venue, link_or_doi, etc.)
SOURCE_MAP: Dict[str, Dict] = {}


def _strip_bibtex_braces(val: str) -> str:
    """Remove BibTeX {} from titles and venues."""
    if not val or not isinstance(val, str):
        return val
    return re.sub(r"[{}]", "", val).strip()


def _parse_authors(authors_str: str) -> List[str]:
    """
    Parse authors string into list of "LastName, Initials" for APA.
    Handles: "Bodnaruc, Alexandra M and Khan, Hassan" or "Author A, Author B, Author C"
    Returns list of (last_name, full_for_ref) for reference formatting.
    Initials do not produce double periods (e.g. "A.. M."); periods are stripped from
    given names before generating initials, then single periods are re-added.
    """
    if not authors_str or not isinstance(authors_str, str):
        return []
    s = authors_str.strip()
    if not s or s.lower() in ("unknown", "n.d.", "tbd"):
        return []

    def _initials_from_words(words: List[str]) -> str:
        """Generate initials (e.g. A. M.) without double periods."""
        if not words:
            return ""
        out = []
        for w in words:
            w = w.strip().rstrip(".")
            if w:
                out.append(w[0].upper() + ".")
        return ". ".join(out)

    # Split on " and " or ";"
    parts = re.split(r"\s+and\s+|\s*;\s*", s, flags=re.IGNORECASE)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Format "LastName, FirstName Middle" -> extract LastName and initials
        if "," in p:
            last, rest = p.split(",", 1)
            last = last.strip()
            words = rest.strip().split()
            init = _initials_from_words(words)
            out.append((last, f"{last}, {init}".strip()))
        else:
            words = p.split()
            if words:
                last = words[-1]
                init = _initials_from_words(words[:-1]) if len(words) > 1 else ""
                ref_part = f"{last}, {init}".strip() if init else last
                out.append((last, ref_part))
    return out


def _load_from_manifest() -> None:
    """Load SOURCE_MAP from manifest.csv."""
    global SOURCE_MAP
    if not MANIFEST_PATH or not Path(MANIFEST_PATH).exists():
        return
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = (row.get("source_id") or "").strip()
                if not sid:
                    continue
                title = _strip_bibtex_braces((row.get("title") or "").strip())
                venue = _strip_bibtex_braces((row.get("venue") or "").strip())
                journal = _strip_bibtex_braces((row.get("venue") or row.get("journal") or "").strip())
                SOURCE_MAP[sid] = {
                    "authors": (row.get("authors") or "").strip(),
                    "year": (row.get("year") or "").strip(),
                    "title": title,
                    "venue": venue,
                    "link_or_doi": (row.get("link_or_doi") or "").strip(),
                    "journal": journal,
                }
    except Exception:
        pass


def _load_from_sources_jsonl() -> None:
    """Merge SOURCE_MAP from sources.jsonl (fill gaps only)."""
    global SOURCE_MAP
    if not SOURCES_JSONL or not Path(SOURCES_JSONL).exists():
        return
    try:
        with open(SOURCES_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = (obj.get("source_id") or "").strip()
                if not sid:
                    continue
                existing = SOURCE_MAP.get(sid, {})
                if not existing.get("authors") and obj.get("authors"):
                    existing = dict(existing)
                    existing["authors"] = (obj.get("authors") or "").strip()
                if not existing.get("year") and obj.get("year"):
                    existing = dict(existing)
                    existing["year"] = str(obj.get("year", "")).strip()
                if not existing.get("title") and obj.get("title"):
                    existing = dict(existing)
                    existing["title"] = _strip_bibtex_braces((obj.get("title") or "").strip())
                if not existing.get("link_or_doi") and (obj.get("doi") or obj.get("url")):
                    existing = dict(existing)
                    d = obj.get("doi") or ""
                    u = obj.get("url") or ""
                    existing["link_or_doi"] = (d if d else u).strip()
                if not existing.get("venue") and obj.get("journal"):
                    existing = dict(existing)
                    existing["venue"] = _strip_bibtex_braces((obj.get("journal") or "").strip())
                SOURCE_MAP[sid] = existing
    except Exception:
        pass


def _ensure_loaded() -> None:
    """Load SOURCE_MAP if not yet loaded."""
    if not SOURCE_MAP:
        _load_from_manifest()
        _load_from_sources_jsonl()


def format_in_text_apa(source_id: str, source_meta: Optional[Dict] = None) -> str:
    """
    APA in-text citation: (Author, Year) or (Author1 & Author2, Year) or (Author et al., Year).
    Uses (Author, n.d.) if year missing. Falls back to source_id if author/year/title unavailable.
    """
    _ensure_loaded()
    meta = source_meta or SOURCE_MAP.get(source_id, {})
    authors_str = meta.get("authors") or ""
    year = (meta.get("year") or "").strip()
    if not year or year.lower() in ("unknown", "tbd", "n.d."):
        year = "n.d."
    parsed = _parse_authors(authors_str)
    if not parsed:
        # Graceful fallback: use source_id as label if author/year/title unavailable (do not crash)
        label = (source_id or "Unknown").replace("_", " ")[:60]
        return f"({label}, {year})"
    last_names = [p[0] for p in parsed]
    if len(last_names) == 1:
        return f"({last_names[0]}, {year})"
    if len(last_names) == 2:
        return f"({last_names[0]} & {last_names[1]}, {year})"
    return f"({last_names[0]} et al., {year})"


def format_reference_apa(source_id: str, source_meta: Optional[Dict] = None) -> str:
    """
    Full APA reference: Author, A. A., Author, B. B., & Author, C. C. (Year). Title. Journal. URL
    Uses DOI as URL when available; else link_or_doi.
    Does NOT include source_id in output.
    """
    _ensure_loaded()
    meta = source_meta or SOURCE_MAP.get(source_id, {})
    authors_str = meta.get("authors") or ""
    year = (meta.get("year") or "").strip()
    if not year or year.lower() in ("unknown", "tbd"):
        year = "n.d."
    title = (meta.get("title") or "Unknown").strip()
    journal = (meta.get("venue") or meta.get("journal") or "").strip()
    link = (meta.get("link_or_doi") or "").strip()
    parsed = _parse_authors(authors_str)
    if not parsed:
        # Graceful fallback: use source_id if author metadata missing
        author_part = (source_id or "Unknown").replace("_", " ")[:60]
    else:
        if len(parsed) == 1:
            author_part = parsed[0][1]
        elif len(parsed) == 2:
            author_part = f"{parsed[0][1]} & {parsed[1][1]}"
        else:
            author_part = ", ".join(p[1] for p in parsed[:-1]) + ", & " + parsed[-1][1]
    ref = f"{author_part} ({year}). {title}."
    if journal:
        ref += f" {journal}."
    if link:
        if not link.startswith("http"):
            link = f"https://doi.org/{link}" if "doi.org" not in link else link
        ref += f" {link}"
    return ref


def get_source_meta(source_id: str) -> Dict:
    """Return metadata dict for source_id. Loads on first access."""
    _ensure_loaded()
    return SOURCE_MAP.get(source_id, {})


def build_internal_to_apa_map(retrieved_chunks: list) -> Dict[str, str]:
    """
    Build chunk_id -> APA in-text citation map for dual citation system.
    Internal format: (chunk_id) e.g. (jmir-2023-1-e37667_c0006)
    APA format: (Author et al., Year)
    """
    _ensure_loaded()
    out = {}
    for c in retrieved_chunks:
        cid = c.get("chunk_id", "")
        sid = c.get("source_id", "")
        if cid and sid:
            out[cid] = format_in_text_apa(sid)
    return out


def map_internal_citations_to_apa(
    answer_text: str,
    internal_to_apa: Dict[str, str],
    valid_chunk_ids: set,
) -> str:
    """
    Replace internal citations (chunk_id) with APA-style in visible answer.
    Only replaces (xxx) when xxx is in valid_chunk_ids and internal_to_apa.
    """
    def replacer(m):
        token = m.group(1)
        if token in valid_chunk_ids and token in internal_to_apa:
            return internal_to_apa[token]
        return m.group(0)

    pattern = re.compile(r"\(([A-Za-z0-9_\-\.]+(?:_[cC]\d+)?)\)")
    return pattern.sub(replacer, answer_text)


def map_internal_to_apa_in_answer_section_only(
    full_output: str,
    internal_to_apa: Dict[str, str],
    valid_chunk_ids: set,
) -> str:
    """
    Replace internal citations with APA only in the Answer section.
    Evidence section keeps internal format for traceability.
    """
    idx = full_output.find("Evidence:")
    if idx == -1:
        return map_internal_citations_to_apa(full_output, internal_to_apa, valid_chunk_ids)
    answer_part = full_output[:idx]
    rest = full_output[idx:]
    answer_mapped = map_internal_citations_to_apa(answer_part, internal_to_apa, valid_chunk_ids)
    return answer_mapped + rest


def build_citation_mapping(retrieved_chunks: list) -> List[dict]:
    """
    Build full citation mapping for logs: [{apa, source_id, chunk_id}, ...]
    For each cited chunk, provides both internal (chunk_id) and APA for traceability.
    """
    _ensure_loaded()
    out = []
    seen = set()
    for c in retrieved_chunks:
        cid = c.get("chunk_id", "")
        sid = c.get("source_id", "")
        if not cid or not sid or cid in seen:
            continue
        seen.add(cid)
        apa = format_in_text_apa(sid)
        out.append({"apa": apa, "source_id": sid, "chunk_id": cid})
    return out


def references_apa_sorted(source_ids: List[str]) -> str:
    """
    Build APA reference list, sorted alphabetically by first author last name.
    Deduplicates by source_id. Returns formatted string with newlines.
    """
    _ensure_loaded()
    seen = set()
    refs = []
    for sid in source_ids:
        if not sid or sid in seen:
            continue
        seen.add(sid)
        refs.append((sid, format_reference_apa(sid), get_source_meta(sid)))
    # Sort by first author last name
    def sort_key(item):
        sid, _, meta = item
        parsed = _parse_authors(meta.get("authors") or "")
        return (parsed[0][0].lower() if parsed else "", meta.get("year", ""), sid)

    refs.sort(key=sort_key)
    if not refs:
        return ""
    return "References:\n" + "\n".join(r[1] for r in refs)


# Load lazily on first format_in_text_apa / format_reference_apa / get_source_meta call
