#!/usr/bin/env python3
"""Build index: ingest -> chunk -> FAISS + BM25. One-command run path."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.config import ensure_dirs
from prp.ingest import ingest_corpus
from prp.chunk import run_chunking
from prp.index import build_all_indexes

if __name__ == "__main__":
    ensure_dirs()
    ingest_corpus()
    run_chunking()
    build_all_indexes()
    print("Index build complete. See data/processed/, indexes/")
