"""Chunking: section-aware optional; fixed size + overlap. Output chunks.jsonl.
Pre-chunk filter removes acknowledgements/funding/conflicts lines (see utils.filter_acknowledgements_and_boilerplate).
"""
import json
from pathlib import Path

from .config import (
    PROCESSED_DIR,
    SOURCES_JSONL,
    CHUNKS_JSONL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ensure_dirs,
)
from .utils import logger, filter_acknowledgements_and_boilerplate, fix_concatenated_words


def chunk_text(
    text: str,
    source_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping chunks. Each chunk: chunk_id, source_id, text, start_char, end_char."""
    chunks = []
    if not text or not text.strip():
        return chunks
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer breaking on paragraph (double newline) or sentence
        if end < len(text):
            search_start = max(start, end - 150)
            chunk_slice = text[search_start:end]
            for sep in ("\n\n", ". ", "\n"):
                pos = chunk_slice.rfind(sep)
                if pos != -1:
                    end = search_start + pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunk_id = f"{source_id}_c{idx:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "source_id": source_id,
                "text": piece,
                "start_char": start,
                "end_char": end,
            })
            idx += 1
        start = end - overlap if (end - overlap) > start else end
    return chunks


def run_chunking(
    sources_path: Path | None = None,
    chunks_path: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Read sources.jsonl, chunk each source, write chunks.jsonl. Returns list of chunk records."""
    ensure_dirs()
    sources_path = sources_path or SOURCES_JSONL
    chunks_path = chunks_path or CHUNKS_JSONL
    if not sources_path.exists():
        raise FileNotFoundError(f"Sources file not found: {sources_path}")

    all_chunks = []
    with open(sources_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            source_id = rec["source_id"]
            text = rec.get("text", "")
            # Fix concatenated words (safety net if ingest normalization missed any)
            text = fix_concatenated_words(text)
            # Remove acknowledgements/funding/conflicts lines before chunking (keeps chunks grounded)
            text = filter_acknowledgements_and_boilerplate(text)
            chunks = chunk_text(text, source_id, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)

    with open(chunks_path, "w", encoding="utf-8") as out:
        for c in all_chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("Chunked %d chunks -> %s (size=%d, overlap=%d)", len(all_chunks), chunks_path, chunk_size, overlap)
    return all_chunks
