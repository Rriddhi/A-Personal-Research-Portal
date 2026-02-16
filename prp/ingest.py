"""Ingestion: discover PDFs from data/raw/ by default; optional source dir (e.g. Research_Papers). Write sources.jsonl and manifest.
When extracting from ZIPs, __MACOSX and similar macOS metadata entries are ignored."""
import csv
import json
import shutil
from pathlib import Path
from typing import Union

from .config import (
    REPO_ROOT,
    RAW_DIR,
    PROCESSED_DIR,
    SOURCES_JSONL,
    MANIFEST_PATH,
    ensure_dirs,
)
from .utils import source_id_from_path, logger, normalize_extracted_text

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# When extracting from ZIPs, skip these (macOS metadata, etc.)
ZIP_SKIP_PREFIXES = ("__MACOSX", ".DS_Store", "._")


def should_skip_zip_member(name: str) -> bool:
    """Return True for ZIP members to ignore (e.g. __MACOSX, .DS_Store)."""
    base = Path(name).parts[0] if name else ""
    return any(base.startswith(p) for p in ZIP_SKIP_PREFIXES)


def _reconstruct_text_from_words(words: list) -> str:
    """Reconstruct page text from word list. Groups by (block, line), joins words with spaces.
    PyMuPDF words: (x0, y0, x1, y1, word, block_no, line_no, word_no)."""
    from collections import defaultdict
    lines_map = defaultdict(list)
    for w in words:
        if len(w) >= 8:
            word_text = w[4]
            block_no, line_no = w[5], w[6]
            x0 = w[0]
            lines_map[(block_no, line_no)].append((x0, word_text))
    # Sort by block, then line; within each line sort by x
    sorted_keys = sorted(lines_map.keys())
    line_texts = []
    for key in sorted_keys:
        items = sorted(lines_map[key], key=lambda x: x[0])
        line_texts.append(" ".join(t for _, t in items))
    return "\n".join(line_texts)


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF word-based extraction for better spacing.
    Uses get_text('words') and reconstructs text to avoid missing spaces between words.
    Returns empty string on failure."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed. pip install PyMuPDF")
    doc = fitz.open(pdf_path)
    try:
        parts = []
        for i in range(len(doc)):
            page = doc[i]
            words = page.get_text("words", sort=True)
            if words:
                parts.append(_reconstruct_text_from_words(words))
            else:
                # Fallback: some PDFs may not return words
                parts.append(page.get_text())
        return "\n\n".join(parts).strip()
    finally:
        doc.close()


def extract_metadata_heuristic(text: str, filename: str) -> dict:
    """Best-effort title/author/year from first ~2000 chars. Placeholders if missing."""
    meta = {"title": "unknown", "authors": "unknown", "year": "unknown"}
    if not text or len(text) < 50:
        return meta
    head = text[:2000]
    # Common pattern: first non-empty line is often title
    lines = [l.strip() for l in head.split("\n") if l.strip()]
    if lines:
        # First substantial line (length > 10) as title placeholder
        for line in lines:
            if len(line) > 10 and not line.isdigit():
                meta["title"] = line[:200]
                break
    # Year: 4-digit 19xx or 20xx
    import re
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", head)
    if years:
        meta["year"] = years[0]
    return meta


def ingest_corpus(force: bool = False, source_dir: Union[Path, str, None] = None) -> list[dict]:
    """
    Discover PDFs from source_dir, write sources.jsonl and manifest.
    Default source_dir is data/raw/. If source_dir is elsewhere, PDFs are copied to data/raw/.
    local_path is always data/raw/<filename>. Returns list of source records.
    """
    ensure_dirs()
    if source_dir is None:
        source_dir = RAW_DIR
    else:
        source_dir = Path(source_dir)
        if not source_dir.is_absolute():
            source_dir = REPO_ROOT / source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", source_dir)

    sources = []
    with open(SOURCES_JSONL, "w", encoding="utf-8") as out:
        for pdf_path in pdfs:
            raw_dest = RAW_DIR / pdf_path.name
            if source_dir.resolve() != RAW_DIR.resolve():
                shutil.copy2(pdf_path, raw_dest)
            # Read from data/raw file so source_id (from raw_dest) is consistent
            read_path = raw_dest
            try:
                text = extract_text_pymupdf(read_path)
            except Exception as e:
                logger.warning("Skipping %s: extraction failed: %s", pdf_path.name, e)
                continue
            # Normalize: fix hyphenation at line breaks, mid-word line breaks, repeated whitespace
            text = normalize_extracted_text(text)
            if not text or len(text) < 50:
                logger.warning("Skipping %s: insufficient text", pdf_path.name)
                continue

            source_id = source_id_from_path(raw_dest)
            local_path = f"data/raw/{pdf_path.name}"
            meta = extract_metadata_heuristic(text, pdf_path.name)
            record = {
                "source_id": source_id,
                "local_path": local_path,
                "title": meta["title"],
                "authors": meta["authors"],
                "year": meta["year"],
                "text": text,
            }
            sources.append(record)
            out.write(json.dumps({**record, "text": text}, ensure_ascii=False) + "\n")

    # Manifest CSV: if manifest exists, load and merge by source_id; preserve existing filled metadata.
    OPERATIONAL_FIELDS = ("local_path", "source_label")  # always from current ingest
    PRESERVE_IF_FILLED = (
        "title", "authors", "year", "type", "venue", "link_or_doi", "relevance_note",
        "bib_key", "needs_manual_review",
    )
    REQUIRED_MANIFEST_COLUMNS = [
        "source_id", "title", "authors", "year", "type", "venue",
        "link_or_doi", "relevance_note", "local_path", "source_label",
    ]
    EMPTY_VALUES = {"", "unknown", "TBD", "Unknown", "tbd", "nan"}

    def _source_label_from_path(local_path: str) -> str:
        stem = Path(local_path).stem
        return stem.replace("-", "_").replace(" ", "_")[:60]

    def _is_filled(v: str) -> bool:
        return bool((v or "").strip()) and (v or "").strip() not in EMPTY_VALUES

    def _new_manifest_row(s: dict) -> dict:
        return {
            "source_id": s["source_id"],
            "title": s["title"],
            "authors": s.get("authors", "unknown"),
            "year": s.get("year", "unknown"),
            "type": "article",
            "venue": "unknown",
            "link_or_doi": "unknown",
            "relevance_note": "TBD",
            "local_path": s["local_path"],
            "source_label": _source_label_from_path(s["local_path"]),
        }

    existing_by_sid = {}
    all_existing_columns = []
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_existing_columns = list(reader.fieldnames or [])
                for row in reader:
                    sid = (row.get("source_id") or "").strip()
                    if sid:
                        existing_by_sid[sid] = dict(row)
        except Exception as e:
            logger.warning("Could not load existing manifest for merge: %s", e)

    ingested_sids = {s["source_id"] for s in sources}
    manifest_rows = []
    for s in sources:
        sid = s["source_id"]
        existing = existing_by_sid.get(sid)
        from_ingest = _new_manifest_row(s)

        if existing:
            # Start from existing row to preserve all columns and values
            new_row = dict(existing)
            # Always update operational fields from current ingest
            new_row["source_id"] = sid
            new_row["local_path"] = from_ingest["local_path"]
            new_row["source_label"] = from_ingest["source_label"]
            # Overwrite metadata only when existing is not filled (keep bib_key, needs_manual_review, etc.)
            for col in PRESERVE_IF_FILLED:
                if col in from_ingest and not _is_filled(existing.get(col)):
                    new_row[col] = from_ingest[col]
        else:
            new_row = from_ingest

        manifest_rows.append(new_row)

    # Keep rows for source_ids no longer in ingest (preserve all columns)
    for sid, row in existing_by_sid.items():
        if sid not in ingested_sids:
            manifest_rows.append(row)

    # Preserve all columns: union of existing header and any keys in merged rows
    all_keys = set(all_existing_columns) if all_existing_columns else set(REQUIRED_MANIFEST_COLUMNS)
    for row in manifest_rows:
        all_keys.update(row.keys())
    all_fieldnames = list(REQUIRED_MANIFEST_COLUMNS) + [k for k in sorted(all_keys - set(REQUIRED_MANIFEST_COLUMNS))]
    if all_existing_columns:
        # Keep original column order where possible: existing order + any new columns at end
        ordered = [c for c in all_existing_columns if c in all_keys]
        ordered += [k for k in sorted(all_keys) if k not in ordered]
        all_fieldnames = ordered
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest_rows)

    logger.info("Ingested %d sources -> %s and %s", len(sources), SOURCES_JSONL, MANIFEST_PATH)
    return sources
