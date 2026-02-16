#!/usr/bin/env python3
"""Run ingestion: discover PDFs from data/raw/ by default; use --source to ingest from another folder (e.g. Research_Papers)."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prp.ingest import ingest_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into data/processed/ and data/manifest.csv. Default source: data/raw/."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source directory for PDFs (default: data/raw/). Use e.g. Research_Papers to copy from there into data/raw/.",
    )
    args = parser.parse_args()
    source_dir = None
    if args.source:
        source_dir = Path(args.source)
        if not source_dir.is_absolute():
            source_dir = REPO / source_dir
    ingest_corpus(source_dir=source_dir)
    print("Ingest complete. Next: make build_index (or make phase2_demo)")


if __name__ == "__main__":
    main()
