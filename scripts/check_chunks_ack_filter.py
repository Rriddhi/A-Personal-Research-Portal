#!/usr/bin/env python3
"""
Quick check: scan data/processed/chunks.jsonl for acknowledgement/funding/conflicts
patterns. After applying the pre-chunk filter and rebuilding, counts should be 0 or near 0.
Run: make ingest && make build_index, then python scripts/check_chunks_ack_filter.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.config import CHUNKS_JSONL
from prp.utils import ACK_FUNDING_PATTERNS

_ACK_RE = re.compile(
    "|".join(re.escape(p) for p in ACK_FUNDING_PATTERNS),
    re.IGNORECASE,
)


def main() -> int:
    if not CHUNKS_JSONL.exists():
        print("No chunks file. Run: make ingest && make build_index")
        return 1
    chunks_with_match = 0
    total_matches = 0
    examples = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "") or ""
            matches = list(_ACK_RE.finditer(text))
            if matches:
                chunks_with_match += 1
                total_matches += len(matches)
                if len(examples) < 3:
                    # Store first matched snippet
                    m = matches[0]
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 50)
                    examples.append((rec.get("chunk_id"), text[start:end].replace("\n", " ")))
    print("Chunks containing ack/funding/conflicts pattern:", chunks_with_match)
    print("Total pattern matches:", total_matches)
    if examples:
        print("Example snippets (first 3):")
        for cid, snippet in examples:
            print(f"  [{cid}] ...{snippet}...")
    if chunks_with_match == 0:
        print("OK: no ack/funding lines in chunks (filter applied successfully).")
    else:
        print("Expected 0 or near 0 after rebuild. Re-run: rm -rf data/processed indexes && make ingest && make build_index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
