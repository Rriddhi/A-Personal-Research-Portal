#!/usr/bin/env python3
"""
Scan data/processed/chunks.jsonl for bad splits.
- Hyphenated line breaks: (\\w)-\\n(\\w) still present (should be joined by normalization).
- Spaced splits: words broken by space, e.g. "in for mation", "per for med", "plat for ms"
  (regex: \\b(in|per|plat)\\s+for\\s+(mation|med|ms)\\b).
After normalization and rebuild, re-run this scan; hyphenated hits should be 0; spaced-split
hits may remain if PDF had mid-word spaces. Run: make ingest && make build_index, then
python scripts/scan_bad_splits.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.config import CHUNKS_JSONL

# Hyphenated line break: word-\nword (should be joined by normalize_extracted_text)
HYPHENATED_LINE_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)", re.MULTILINE)

# Spaced splits: "in for mation", "per for med", "plat for ms"
SPACED_SPLIT_RE = re.compile(
    r"\b(in|per|plat)\s+for\s+(mation|med|ms)\b",
    re.IGNORECASE,
)


def main() -> int:
    if not CHUNKS_JSONL.exists():
        print("No chunks file. Run: make ingest && make build_index")
        return 1

    hyphen_hits = []
    spaced_hits = []

    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id", "")
            sid = rec.get("source_id", "")
            text = rec.get("text", "") or ""

            for m in HYPHENATED_LINE_BREAK_RE.finditer(text):
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 40)
                excerpt = text[start:end].replace("\n", " ")
                hyphen_hits.append((cid, sid, excerpt))

            for m in SPACED_SPLIT_RE.finditer(text):
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                excerpt = text[start:end].replace("\n", " ")
                spaced_hits.append((cid, sid, excerpt))

    print("Bad-splits scan results:")
    print("  Hyphenated line breaks (word-\\nword):", len(hyphen_hits))
    print("  Spaced splits (in/per/plat + for + mation/med/ms):", len(spaced_hits))

    if hyphen_hits:
        print("\nHyphenated line-break examples (first 5):")
        for cid, sid, excerpt in hyphen_hits[:5]:
            print(f"  [{cid}] {sid}: ...{excerpt}...")
    if spaced_hits:
        print("\nSpaced-split examples (first 5):")
        for cid, sid, excerpt in spaced_hits[:5]:
            print(f"  [{cid}] {sid}: ...{excerpt}...")

    if not hyphen_hits and not spaced_hits:
        print("OK: no bad splits found.")
        return 0
    if hyphen_hits:
        print("\nRe-run normalization and rebuild: make ingest && make build_index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
