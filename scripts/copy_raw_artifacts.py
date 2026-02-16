#!/usr/bin/env python3
"""Copy PDFs from Research_Papers/ to data/raw/ and update manifest (local_path, source_label)."""
import csv
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.config import RESEARCH_PAPERS_DIR, RAW_DIR, DATA_DIR, MANIFEST_PATH, ensure_dirs


def _source_label_from_filename(filename: str) -> str:
    """Human-readable short id from PDF filename (e.g. jmir-2023-1-e37667.pdf -> jmir_2023_1_e37667)."""
    stem = Path(filename).stem
    return stem.replace("-", "_").replace(" ", "_")[:60]


def main():
    ensure_dirs()
    if not RESEARCH_PAPERS_DIR.exists():
        print(f"Corpus directory not found: {RESEARCH_PAPERS_DIR}")
        return 1

    pdfs = sorted(RESEARCH_PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in", RESEARCH_PAPERS_DIR)
        return 1

    for pdf_path in pdfs:
        dest = RAW_DIR / pdf_path.name
        shutil.copy2(pdf_path, dest)
        print("Copied", pdf_path.name, "->", dest)

    if not MANIFEST_PATH.exists():
        print("Manifest not found; run ingest first.")
        return 0

    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "source_label" not in fieldnames:
            fieldnames = list(fieldnames) + ["source_label"]
        for row in reader:
            old_path = row.get("local_path", "")
            basename = Path(old_path).name
            row["local_path"] = f"data/raw/{basename}"
            row["source_label"] = row.get("source_label") or _source_label_from_filename(basename)
            rows.append(row)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("Updated manifest: local_path -> data/raw/<filename>, added source_label")
    return 0


if __name__ == "__main__":
    sys.exit(main())
