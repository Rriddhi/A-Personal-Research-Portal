#!/usr/bin/env python3
"""Detect duplicate PDFs in data/raw by file hash. Reports duplicates and exits with 1 if any found."""
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "data" / "raw"


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(raw_dir: Path) -> dict:
    """Return hash -> list of file paths (duplicates only)."""
    hashes: dict[str, list[Path]] = {}
    if not raw_dir.exists():
        return {}
    for pdf in sorted(raw_dir.glob("*.pdf")):
        try:
            h = file_hash(pdf)
            hashes.setdefault(h, []).append(pdf)
        except OSError as e:
            print(f"Warning: could not read {pdf}: {e}", file=sys.stderr)
    return {h: paths for h, paths in hashes.items() if len(paths) > 1}


def main() -> int:
    raw_dir = RAW_DIR
    if len(sys.argv) > 1:
        raw_dir = Path(sys.argv[1])
    dups = find_duplicates(raw_dir)
    if not dups:
        print(f"No duplicate PDFs found in {raw_dir}")
        return 0
    print(f"Duplicate PDFs in {raw_dir}:")
    for h, paths in dups.items():
        print(f"  Hash {h[:16]}...:")
        for p in paths:
            print(f"    - {p.name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
