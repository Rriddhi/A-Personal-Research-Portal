#!/usr/bin/env python3
"""
Update manifest rows with needs_manual_review=1 by extracting title/year/venue/DOI
from the first page of each PDF at local_path. Set needs_manual_review=0 when
title, year, and link_or_doi are all filled; otherwise leave 1 and add a note to relevance_note.
Requires PyMuPDF (pip install PyMuPDF) for PDF text extraction.
"""
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

EMPTY_VALUES = {"", "unknown", "TBD", "Unknown"}
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
DOI_URL_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

VENUE_PATTERNS = [
    (r"J\.?\s*Med\.?\s*Internet\s*Res", "JMIR"),
    (r"Nature\s*Medicine", "Nature Medicine"),
    (r"npj\s*[|\|]\s*digital\s*medicine", "npj Digital Medicine"),
    (r"npj\s*health\s*systems", "npj Health Systems"),
    (r"Frontiers\s+in\s+Public\s+Health", "Frontiers in Public Health"),
    (r"Frontiers\s+in\s+\w+", "Frontiers"),
    (r"International\s+Journal\s+of\s+Medical\s+Informatics", "IJMI"),
    (r"American\s+Journal\s+of\s+Clinical\s+Nutrition", "AJCN"),
    (r"communications\s+medicine", "Communications Medicine"),
    (r"Scientific\s+Reports", "Scientific Reports"),
    (r"Clinical\s+Nutrition\s+ESPEN", "Clinical Nutrition ESPEN"),
    (r"Advances\s+in\s+Nutrition", "Advances in Nutrition"),
    (r"BMC\s+", "BMC"),
    (r"Elsevier", "Elsevier"),
    (r"Springer", "Springer"),
]


def _is_empty(val: str) -> bool:
    if val is None:
        return True
    s = (val or "").strip()
    return s in EMPTY_VALUES or not s


def extract_first_page_text(pdf_path: Path) -> str:
    """Extract text from first page of PDF using PyMuPDF. Returns empty string on failure."""
    try:
        import fitz
    except ImportError:
        return ""
    if not pdf_path.exists():
        return ""
    try:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text() if len(doc) else ""
        doc.close()
        return text.strip()
    except Exception:
        return ""


def infer_doi(text: str) -> str:
    m = DOI_URL_RE.search(text)
    if m:
        return m.group(1).strip()
    m = DOI_RE.search(text)
    if m:
        return m.group(0).strip()
    return ""


def infer_year(text: str) -> str:
    head = text[:3000]
    for match in YEAR_RE.findall(head):
        yy = int(match)
        if 1990 <= yy <= 2030:
            return match
    return ""


def infer_title(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
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
    best = ""
    for c in candidates:
        if 20 <= len(c) <= 250 and "http" not in c and "doi.org" not in c:
            if len(c) > len(best):
                best = c
    return best or (candidates[0] if candidates else "")


def infer_venue(text: str) -> str:
    for pattern, name in VENUE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return ""


def infer_link_or_doi(text: str) -> str:
    doi = infer_doi(text)
    if doi:
        return f"https://doi.org/{doi}" if not doi.startswith("http") else doi
    return ""


def main() -> int:
    if not MANIFEST_PATH.exists():
        print("Manifest not found:", MANIFEST_PATH, file=sys.stderr)
        return 1

    rows = []
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    # Ensure columns exist
    for col in ("needs_manual_review", "relevance_note"):
        if col not in fieldnames:
            fieldnames.append(col)

    to_review = [r for r in rows if str(r.get("needs_manual_review", "")).strip() == "1"]
    if not to_review:
        print("No rows with needs_manual_review=1. Nothing to do.")
        return 0

    print(f"Processing {len(to_review)} rows with needs_manual_review=1...")
    updated = 0
    still_need_review = 0

    for row in to_review:
        local_path = (row.get("local_path") or "").strip()
        if not local_path:
            row["needs_manual_review"] = "1"
            row["relevance_note"] = (row.get("relevance_note") or "TBD") + " [Missing: local_path]"
            still_need_review += 1
            continue

        pdf_path = REPO / local_path
        if not pdf_path.exists():
            row["needs_manual_review"] = "1"
            row["relevance_note"] = (row.get("relevance_note") or "TBD") + " [PDF not found at local_path]"
            still_need_review += 1
            continue

        text = extract_first_page_text(pdf_path)
        if not text or len(text) < 50:
            row["needs_manual_review"] = "1"
            row["relevance_note"] = (row.get("relevance_note") or "TBD") + " [Could not extract text from first page]"
            still_need_review += 1
            continue

        # Fill only when current value is empty/unknown
        if _is_empty(row.get("title")):
            row["title"] = infer_title(text) or row.get("title") or "unknown"
        if _is_empty(row.get("year")):
            row["year"] = infer_year(text) or row.get("year") or "unknown"
        if _is_empty(row.get("venue")):
            row["venue"] = infer_venue(text) or row.get("venue") or "unknown"
        if _is_empty(row.get("link_or_doi")):
            row["link_or_doi"] = infer_link_or_doi(text) or row.get("link_or_doi") or "unknown"

        # Clear needs_manual_review if title, year, link_or_doi all filled
        title_ok = not _is_empty(row.get("title"))
        year_ok = not _is_empty(row.get("year"))
        doi_ok = not _is_empty(row.get("link_or_doi"))

        if title_ok and year_ok and doi_ok:
            row["needs_manual_review"] = "0"
            updated += 1
        else:
            row["needs_manual_review"] = "1"
            missing = []
            if not title_ok:
                missing.append("title")
            if not year_ok:
                missing.append("year")
            if not doi_ok:
                missing.append("DOI/link")
            note = (row.get("relevance_note") or "TBD").strip()
            if note in ("TBD", ""):
                note = "This source discusses personalized nutrition and health and is relevant for evidence-grounded decision support."
            row["relevance_note"] = note + " [Missing: " + ", ".join(missing) + "]"
            still_need_review += 1

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Updated manifest: {updated} rows set to needs_manual_review=0, {still_need_review} still need manual review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
