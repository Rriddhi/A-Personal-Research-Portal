#!/usr/bin/env python3
"""
Regression test: extraction normalization fixes mid-word line breaks and hyphenation.
Asserts that chunks from jmir-2023-1-e37667 contain expected semantic content (anchor phrases)
and do not contain mangled tokens like "ilored".
Requires: make ingest && make build_index (chunks.jsonl must exist).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.config import CHUNKS_JSONL

# Source with known normalization issues (e.g. "tai\nlored" -> "tailored")
SOURCE_STEM = "jmir-2023-1-e37667"

# Semantic anchors: at least one must appear somewhere in the source's chunks
# (survives chunk boundary drift; equivalent to tailored/personalized nutrition passage)
ANCHOR_PHRASES = [
    "tailored",
    "generic advice",
    "personal data",
    "preferences",
    "evidence-based dietary",
    "individual",
]


def main() -> int:
    if not CHUNKS_JSONL.exists():
        print("SKIP: chunks.jsonl not found. Run: make ingest && make build_index")
        return 0

    chunks_from_source = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id", "")
            if SOURCE_STEM in cid:
                text = (rec.get("text") or "")
                chunks_from_source.append((cid, text))

    if not chunks_from_source:
        print("SKIP: No chunks from %s in %s" % (SOURCE_STEM, CHUNKS_JSONL))
        return 0

    # No chunk should contain broken "ilored" as standalone word (not as part of "tailored")
    import re
    broken_ilored = re.compile(r"\bilored\b")
    for cid, text in chunks_from_source:
        assert not broken_ilored.search(text), (
            "Chunk %s should not contain broken 'ilored'; normalization should merge mid-word lines. "
            "Excerpt: ...%s..." % (cid, text[:200])
        )

    # At least one chunk should contain a semantic anchor (survives boundary drift)
    for cid, text in chunks_from_source:
        for anchor in ANCHOR_PHRASES:
            if anchor in text:
                print("PASS: Chunk %s contains '%s' and no chunk contains 'ilored'." % (cid, anchor))
                return 0

    excerpt = chunks_from_source[0][1][:200] if chunks_from_source else ""
    assert False, (
        "No chunk from %s contains any anchor phrase. Expected one of: %s. "
        "Example chunk_id=%s, excerpt: ...%s..."
        % (SOURCE_STEM, ANCHOR_PHRASES, chunks_from_source[0][0], excerpt)
    )


if __name__ == "__main__":
    sys.exit(main())
