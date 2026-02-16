#!/usr/bin/env python3
"""Enrich manifest metadata from first 1-2 pages of each PDF in data/raw/. Best-effort, no API."""
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MANIFEST_ENRICHED_PATH = DATA_DIR / "manifest_enriched.csv"

# Values we treat as "empty" and may overwrite
EMPTY_VALUES = {"", "unknown", "TBD", "Unknown"}

# DOI pattern (Crossref-style); also match after doi.org/
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
DOI_URL_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)

# Year in title/header area
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Venue patterns (order matters: more specific first)
VENUE_PATTERNS = [
    (r"J\.?\s*Med\.?\s*Internet\s*Res", "JMIR"),
    (r"Nature\s*Medicine", "Nature Medicine"),
    (r"npj\s*[|\|]\s*digital\s*medicine", "npj Digital Medicine"),
    (r"npj\s*health\s*systems", "npj Health Systems"),
    (r"Frontiers\s+in\s+Public\s+Health", "Frontiers in Public Health"),
    (r"Frontiers\s+in\s+\w+", "Frontiers"),
    (r"arXiv", "arXiv"),
    (r"IEEE\s+", "IEEE"),
    (r"ACM\s+", "ACM"),
    (r"Elsevier", "Elsevier"),
    (r"Springer", "Springer"),
    (r"BMC\s+", "BMC"),
    (r"American\s+Journal\s+of\s+Clinical\s+Nutrition", "AJCN"),
    (r"International\s+Journal\s+of\s+Medical\s+Informatics", "IJMI"),
    (r"communications\s+medicine", "Communications Medicine"),
    (r"Cureus", "Cureus"),
    (r"formative", "Formative"),
    (r"metabolites", "Metabolites"),
    (r"cancer", "Cancer"),
    (r"dietary\s+guidelines", "Dietary Guidelines"),
]

# Keywords for relevance note (deterministic topic extraction)
RELEVANCE_KEYWORDS = [
    "personalized", "nutrition", "dietary", "health", "AI", "artificial intelligence",
    "precision", "recommendation", "evidence", "systematic review", "randomized",
    "patient", "clinical", "guidelines", "decision support", "recommender",
]


def extract_first_pages_text(pdf_path: Path, max_pages: int = 2) -> str:
    """Extract text from first max_pages of PDF using PyMuPDF. Returns empty string on failure."""
    try:
        import fitz
    except ImportError:
        return ""
    if not pdf_path.exists():
        return ""
    try:
        doc = fitz.open(pdf_path)
        parts = []
        for i in range(min(max_pages, len(doc))):
            parts.append(doc[i].get_text())
        doc.close()
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def _is_empty(val: str) -> bool:
    if val is None:
        return True
    s = (val or "").strip()
    return s in EMPTY_VALUES or not s


def infer_doi(text: str) -> str:
    m = DOI_RE.search(text)
    if m:
        return m.group(0).strip()
    m2 = DOI_URL_RE.search(text)
    if m2:
        return m2.group(1).strip()
    return ""


def infer_year(text: str) -> str:
    """First 20xx/19xx in first ~3000 chars (title/header area)."""
    head = text[:3000]
    matches = YEAR_RE.findall(head)
    for y in matches:
        yy = int(y)
        if 1990 <= yy <= 2030:
            return str(y)
    if matches:
        return str(matches[0])
    return ""


def infer_title(text: str) -> str:
    """Largest/most prominent non-boilerplate line, or first long line before Abstract."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Drop common boilerplate
    skip = {"article", "original research", "review", "open access", "doi:", "http", "www.", "©", "published", "received", "accepted"}
    candidates = []
    for ln in lines:
        if len(ln) < 10:
            continue
        lower = ln.lower()
        if any(lower.startswith(s) or s in lower[:20] for s in skip):
            continue
        if "abstract" in lower and len(ln) < 100:
            break
        candidates.append(ln)
    if not candidates:
        return ""
    # Prefer longest line that looks like a title (not a URL, not too long)
    best = ""
    for c in candidates:
        if 20 <= len(c) <= 250 and "http" not in c and "doi.org" not in c:
            if len(c) > len(best):
                best = c
    return best or (candidates[0] if candidates else "")


def infer_authors(text: str) -> str:
    """Heuristic: line(s) with multiple comma-separated parts after title area."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:15]:
        if len(ln) < 5 or len(ln) > 400:
            continue
        if "abstract" in ln.lower() or "doi" in ln.lower() or "http" in ln.lower():
            break
        parts = [p.strip() for p in re.split(r"[,;]", ln) if p.strip()]
        if 2 <= len(parts) <= 15 and all(3 <= len(p) <= 50 for p in parts[:5]):
            return ln[:400]
    return ""


def infer_venue(text: str) -> str:
    for pattern, name in VENUE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return ""


def infer_type(text: str) -> str:
    lower = text[:4000].lower()
    if "systematic review" in lower or "scoping review" in lower:
        return "review"
    if "randomized" in lower and ("trial" in lower or "controlled" in lower):
        return "RCT"
    if "technical report" in lower or "guidance" in lower:
        return "technical report"
    if "standard" in lower and ("iso" in lower or "ieee" in lower):
        return "standard"
    return "paper"


def infer_relevance_note(text: str) -> str:
    """Deterministic 1-2 sentence note from keywords only. No hallucination."""
    lower = text[:5000].lower()
    found = [kw for kw in RELEVANCE_KEYWORDS if kw.lower() in lower]
    topic = ", ".join(found[:3]) if found else "personalized nutrition and health"
    sub = ", ".join(found[3:6]) if len(found) > 3 else "evidence and recommendations"
    return (
        f"This source discusses {topic} and is relevant because it informs {sub} "
        "in evidence-grounded personalized nutrition/health decision support."
    )


def infer_link_or_doi(text: str) -> str:
    doi = infer_doi(text)
    if doi:
        return f"https://doi.org/{doi}" if not doi.startswith("http") else doi
    return ""


def enrich_row(row: dict, pdf_path: Path) -> tuple[dict, bool]:
    """Enrich one row using PDF text. Returns (updated_row, needs_manual_review). Only overwrites blank/unknown/TBD."""
    text = extract_first_pages_text(pdf_path, max_pages=2)
    needs_review = 0

    if _is_empty(row.get("title")):
        v = infer_title(text)
        row["title"] = v if v else (row.get("title") or "unknown")
        if not v:
            needs_review = 1
    if _is_empty(row.get("year")):
        v = infer_year(text)
        row["year"] = v if v else (row.get("year") or "unknown")
        if not v:
            needs_review = 1
    if _is_empty(row.get("authors")):
        v = infer_authors(text)
        row["authors"] = v if v else (row.get("authors") or "unknown")
    if _is_empty(row.get("link_or_doi")):
        v = infer_link_or_doi(text)
        row["link_or_doi"] = v if v else (row.get("link_or_doi") or "unknown")
        if not v:
            needs_review = 1
    if _is_empty(row.get("venue")):
        row["venue"] = infer_venue(text) or (row.get("venue") or "unknown")
    if _is_empty(row.get("type")) or (row.get("type") or "").lower() == "article":
        row["type"] = infer_type(text) or (row.get("type") or "paper")
    if _is_empty(row.get("relevance_note")) or (row.get("relevance_note") or "").strip() == "TBD":
        row["relevance_note"] = infer_relevance_note(text)

    row["needs_manual_review"] = 1 if needs_review else 0
    return row, bool(needs_review)


def main() -> int:
    if not MANIFEST_PATH.exists():
        print("Manifest not found:", MANIFEST_PATH)
        return 1

    rows = []
    fieldnames = None
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "needs_manual_review" not in fieldnames:
            fieldnames.append("needs_manual_review")
        for row in reader:
            rows.append(row)

    # Resolve local_path relative to repo
    enriched = []
    needs_review_list = []
    for row in rows:
        local_path = row.get("local_path", "").strip()
        if not local_path:
            enriched.append(row)
            row["needs_manual_review"] = 1
            needs_review_list.append(row.get("source_id", ""))
            continue
        pdf_path = (REPO / local_path) if not Path(local_path).is_absolute() else Path(local_path)
        new_row, need = enrich_row(dict(row), pdf_path)
        if need:
            needs_review_list.append(new_row.get("source_id", ""))
        enriched.append(new_row)

    # Write enriched manifest
    MANIFEST_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_ENRICHED_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(enriched)

    # Summary
    still_unknown_title = sum(1 for r in enriched if _is_empty(r.get("title")))
    still_unknown_year = sum(1 for r in enriched if _is_empty(r.get("year")))
    still_unknown_doi = sum(1 for r in enriched if _is_empty(r.get("link_or_doi")))
    n_review = len(needs_review_list)

    print("--- Enrichment summary ---")
    print("Rows processed:", len(enriched))
    print("Rows with at least one field enriched: all (best-effort)")
    print("Still unknown title:", still_unknown_title)
    print("Still unknown year:", still_unknown_year)
    print("Still unknown DOI/link:", still_unknown_doi)
    print("Needing manual review (needs_manual_review=1):", n_review)
    if needs_review_list:
        print("Sources needing manual review:")
        for sid in needs_review_list[:20]:
            print("  -", sid)
        if len(needs_review_list) > 20:
            print("  ... and", len(needs_review_list) - 20, "more")

    # Diff summary: overwrite manifest.csv with enriched
    print("\nEnriched manifest written to:", MANIFEST_ENRICHED_PATH)
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(enriched)
    print("Updated data/manifest.csv with enriched metadata.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
