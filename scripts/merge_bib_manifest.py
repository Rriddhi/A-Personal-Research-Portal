#!/usr/bin/env python3
"""
Merge Zotero BibTeX (library.bib) metadata into data/manifest.csv without changing source_id.
Outputs data/manifest_from_bib.csv (preview); use --write to overwrite data/manifest.csv.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
PREVIEW_PATH = DATA_DIR / "manifest_from_bib.csv"

# Values we treat as empty and may overwrite from BibTeX
EMPTY_VALUES = {"", "unknown", "TBD", "Unknown", "unknown ", "tbd", "nan"}

# Placeholder titles (PDF boilerplate) — treat as missing so BibTeX can overwrite
PLACEHOLDER_TITLES = {
    "Original Research Article",
    "Original article",
    "Original Paper",
    "Open Access",
    "PERSPECTIVE",
    "Publications",
    "Vol.:(0123456789)",
}

# DOI patterns
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
DOI_URL_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)

# source_id prefixes: Nature (10.1038), BMC (10.1186), or title match
PREFIX_DOI_NATURE = ("s41591-", "s41746-", "s41598-", "s43856-", "s44401-")
PREFIX_DOI_BMC = ("s12911-", "s41512-")
PREFIX_TITLE_MATCH = ("jmir-", "formative-", "cancer-", "fdgth-", "fpubh-", "jpm-", "metabolites-")


def _is_empty(val: str) -> bool:
    if val is None:
        return True
    s = (val or "").strip()
    return s in EMPTY_VALUES or not s


def is_missing(val: str, field: str = None) -> bool:
    """
    True when value is empty/placeholder so BibTeX can overwrite.
    For title/venue/authors/link_or_doi: empty, NaN, unknown, tbd/TBD, and (title only) placeholder titles.
    """
    if val is None:
        return True
    s = (str(val).strip() if val == val else "").strip()  # avoid NaN
    if not s or s.lower() in ("unknown", "tbd", "nan"):
        return True
    if field == "title" and s in PLACEHOLDER_TITLES:
        return True
    return False


def normalize_doi(doi: str) -> str:
    """Lowercase, strip; extract from URL if needed."""
    if not doi or not (doi := str(doi).strip()):
        return ""
    m = DOI_URL_RE.search(doi)
    if m:
        return m.group(1).lower().strip()
    m = DOI_RE.search(doi)
    if m:
        return m.group(0).lower().strip()
    return doi.lower().strip()


def _base_from_row(row: dict) -> str:
    """Filename base (no .pdf, no hash) from local_path or source_id."""
    path = (row.get("local_path") or "").strip()
    if path:
        base = Path(path).stem
        if base:
            return base
    sid = (row.get("source_id") or "").strip()
    if sid:
        # source_id is like "s41591-024-02951-6.pdf_1ecb7a43c514" or "jmir-2023-1-e37667.pdf_xxx"
        part = sid.split("._")[0].split(".pdf_")[0].strip()
        if part.endswith(".pdf"):
            part = part[:-4]
        if part:
            return part
    return ""


def extract_doi_from_manifest_row(row: dict) -> str:
    """Get DOI from link_or_doi or from filename/source_id when possible."""
    link = (row.get("link_or_doi") or "").strip()
    if link and link not in EMPTY_VALUES:
        d = normalize_doi(link)
        if d:
            return d
    base = _base_from_row(row)
    if not base:
        return ""
    # BMC first (s12911, s41512)
    for prefix in PREFIX_DOI_BMC:
        if base.startswith(prefix):
            return "10.1186/" + base
    # Nature: s41591-*, s41598-*, s41746-*, s43856-*, s44401-*
    for prefix in PREFIX_DOI_NATURE:
        if base.startswith(prefix):
            return "10.1038/" + base
    # JMIR / Formative / Cancer: jmir-2023-1-e37667 -> 10.2196/37667
    if "jmir-" in base and "e" in base:
        for p in base.split("-"):
            if p.startswith("e") and len(p) > 1 and p[1:].isdigit():
                return "10.2196/" + p[1:]
    if "formative-" in base and "e" in base:
        for p in base.split("-"):
            if p.startswith("e") and len(p) > 1 and p[1:].isdigit():
                return "10.2196/" + p[1:]
    if "cancer-" in base and "e" in base:
        for p in base.split("-"):
            if p.startswith("e") and len(p) > 1 and p[1:].isdigit():
                return "10.2196/" + p[1:]
    # Frontiers: fdgth-07-1692517 -> 10.3389/fdgth.2025.1692517, fpubh-13-1689911 -> 10.3389/fpubh.2025.1689911
    if base.startswith("fdgth-"):
        parts = base.split("-")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "10.3389/fdgth.2025." + parts[-1]
    if base.startswith("fpubh-"):
        parts = base.split("-")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "10.3389/fpubh.2025." + parts[-1]
    # jpm-15-00058, metabolites-15-00653 (single known DOI pattern)
    if base.startswith("jpm-"):
        return "10.3390/jpm15020058"
    if base.startswith("metabolites-"):
        return "10.3390/metabo15100653"
    return ""


def _read_bib_value(s: str, start: int) -> tuple[str, int]:
    """From s[start], read balanced-brace value after '=', return (value, end_index)."""
    i = start
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i >= len(s) or s[i] != "{":
        return "", start
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j].strip(), j + 1
        j += 1
    return "", start


def parse_bib_file(bib_path: Path) -> list[dict]:
    """Parse .bib file into list of entries with key, doi, title, author, year, journal."""
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries = []
    # Split by @type{key,
    pattern = re.compile(r"@\s*(\w+)\s*\{\s*([^,]+)\s*,")
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            break
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        block_start = m.end()
        # Find end of this entry (next @ at line start or end of file)
        next_at = text.find("\n@", block_start)
        if next_at == -1:
            block = text[block_start:]
        else:
            block = text[block_start : next_at + 1]
        pos = block_start + len(block)

        # Only articles (and misc/inbook if they have doi) for matching
        if entry_type not in ("article", "inproceedings", "conference", "misc", "book"):
            continue

        fields = {}
        for field_name in ("doi", "title", "author", "year", "journal", "url"):
            rx = re.compile(r"\b" + field_name + r"\s*=", re.IGNORECASE)
            fm = rx.search(block)
            if fm:
                val, _ = _read_bib_value(block, fm.end())
                if val:
                    fields[field_name.lower()] = val
        if not fields:
            continue
        entries.append({
            "key": key,
            "doi": normalize_doi(fields.get("doi", "")),
            "title": fields.get("title", "").strip(),
            "author": fields.get("author", "").strip(),
            "year": (fields.get("year", "") or "").strip(),
            "journal": fields.get("journal", "").strip(),
            "url": fields.get("url", "").strip(),
        })
    return entries


def build_bib_by_doi(entries: list[dict]) -> dict[str, dict]:
    """Map normalized DOI -> first bib entry."""
    by_doi = {}
    for e in entries:
        if e["doi"]:
            by_doi.setdefault(e["doi"], e)
    return by_doi


def title_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity in [0,1]. Normalize: lower, alphanumeric tokens."""
    def tokens(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def find_bib_by_title(entries: list[dict], manifest_title: str, source_id: str) -> dict | None:
    """Best bib entry by title similarity for title-match prefixes."""
    if not manifest_title or _is_empty(manifest_title):
        return None
    best = None
    best_score = 0.0
    for e in entries:
        if not e.get("title"):
            continue
        score = title_similarity(manifest_title, e["title"])
        if score > best_score and score >= 0.25:
            best_score = score
            best = e
    return best


def match_manifest_row_to_bib(row: dict, bib_by_doi: dict[str, dict], bib_entries: list[dict]) -> dict | None:
    """
    Return matched bib entry or None. Do not change source_id or local_path.
    Matching: (a) DOI from row or derived from filename (b) title similarity for jmir/formative/cancer etc.
    """
    source_id = (row.get("source_id") or "").strip()
    # 1) DOI match (from link_or_doi or derived from filename)
    doi = extract_doi_from_manifest_row(row)
    if doi and doi in bib_by_doi:
        return bib_by_doi[doi]

    # 2) Title similarity for known prefixes (when DOI match failed)
    sid_lower = source_id.lower()
    for prefix in PREFIX_TITLE_MATCH:
        if sid_lower.startswith(prefix.replace("-", "_")) or prefix in sid_lower or source_id.startswith(prefix):
            manifest_title = (row.get("title") or "").strip()
            return find_bib_by_title(bib_entries, manifest_title, source_id)

    for prefix in PREFIX_DOI_NATURE + PREFIX_DOI_BMC:
        if source_id.startswith(prefix) or prefix.replace("-", "_") in sid_lower:
            manifest_title = (row.get("title") or "").strip()
            return find_bib_by_title(bib_entries, manifest_title, source_id)

    return None


def merge_row(row: dict, bib: dict | None) -> dict:
    """Fill title, authors, year, venue, link_or_doi from bib when current is empty. Add bib_key. Set needs_manual_review."""
    out = dict(row)
    if "bib_key" not in out:
        out["bib_key"] = ""
    if "needs_manual_review" not in out:
        out["needs_manual_review"] = "1"

    if not bib:
        # No match: ensure needs_manual_review=1
        out["needs_manual_review"] = "1"
        return out

    out["bib_key"] = bib.get("key", "")
    if is_missing(out.get("title"), "title") and bib.get("title"):
        out["title"] = bib["title"]
    if is_missing(out.get("authors"), "authors") and bib.get("author"):
        out["authors"] = bib["author"]
    if is_missing(out.get("year"), "year") and bib.get("year"):
        out["year"] = bib["year"]
    if is_missing(out.get("venue"), "venue") and bib.get("journal"):
        out["venue"] = bib["journal"]
    if is_missing(out.get("link_or_doi"), "link_or_doi"):
        if bib.get("doi"):
            out["link_or_doi"] = "https://doi.org/" + bib["doi"]
        elif bib.get("url"):
            out["link_or_doi"] = bib["url"]
        else:
            out["link_or_doi"] = out.get("link_or_doi") or "unknown"

    # needs_manual_review: 1 if title/year/link_or_doi still missing OR no bib match
    if is_missing(out.get("title"), "title") or is_missing(out.get("year"), "year") or is_missing(out.get("link_or_doi"), "link_or_doi"):
        out["needs_manual_review"] = "1"
    else:
        out["needs_manual_review"] = "0"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge BibTeX metadata into manifest (preview or --write).")
    parser.add_argument("--bib", default=None, help="Path to .bib file (default: library.bib or My Library.bib in repo root)")
    parser.add_argument("--write", action="store_true", help="Overwrite data/manifest.csv with merged result")
    args = parser.parse_args()

    bib_path = None
    if args.bib:
        bib_path = Path(args.bib)
        if not bib_path.is_absolute():
            bib_path = REPO / bib_path
    else:
        for name in ("library.bib", "My Library.bib"):
            cand = REPO / name
            if cand.exists():
                bib_path = cand
                break
    if not bib_path or not bib_path.exists():
        print("Error: Bib file not found. Use --bib path or place library.bib / My Library.bib in repo root.", file=sys.stderr)
        sys.exit(1)

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read manifest
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "bib_key" not in fieldnames:
        fieldnames.append("bib_key")
    if "needs_manual_review" not in fieldnames:
        fieldnames.append("needs_manual_review")

    # Parse bib
    bib_entries = parse_bib_file(bib_path)
    bib_by_doi = build_bib_by_doi(bib_entries)

    # Match and merge
    merged = []
    matched = 0
    fields_updated = {"title": 0, "authors": 0, "year": 0, "venue": 0, "link_or_doi": 0}
    not_matched_ids = []

    for row in rows:
        bib = match_manifest_row_to_bib(row, bib_by_doi, bib_entries)
        if bib:
            matched += 1
        else:
            not_matched_ids.append(row.get("source_id", ""))

        out = merge_row(row, bib)
        merged.append(out)
        # Count fields that were missing and are now filled from BibTeX
        if is_missing(row.get("title"), "title") and not is_missing(out.get("title"), "title"):
            fields_updated["title"] += 1
        if is_missing(row.get("authors"), "authors") and not is_missing(out.get("authors"), "authors"):
            fields_updated["authors"] += 1
        if is_missing(row.get("year"), "year") and not is_missing(out.get("year"), "year"):
            fields_updated["year"] += 1
        if is_missing(row.get("venue"), "venue") and not is_missing(out.get("venue"), "venue"):
            fields_updated["venue"] += 1
        if is_missing(row.get("link_or_doi"), "link_or_doi") and not is_missing(out.get("link_or_doi"), "link_or_doi"):
            fields_updated["link_or_doi"] += 1

    # Write preview
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREVIEW_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)

    # Summary
    print("Summary:")
    print(f"  total rows: {len(rows)}")
    print(f"  matched rows: {matched}")
    print("  fields updated (count of rows where field was empty and filled from BibTeX):")
    for k, v in fields_updated.items():
        print(f"    {k}: {v}")
    print("  source_ids not matched (needs_manual_review=1):")
    for sid in not_matched_ids:
        print(f"    {sid}")

    if args.write:
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)
        print(f"\nWrote {MANIFEST_PATH}")
    else:
        print(f"\nPreview written to {PREVIEW_PATH} (use --write to overwrite manifest.csv)")


if __name__ == "__main__":
    main()
