"""Index building: FAISS vector index + docstore + BM25 for hybrid retrieval."""
import json
from pathlib import Path
from typing import List

import numpy as np

from .config import (
    CHUNKS_JSONL,
    INDEXES_DIR,
    FAISS_INDEX_PATH,
    DOCSTORE_PATH,
    BM25_DIR,
    ensure_dirs,
)
from .embed import embed_texts
from .utils import logger

# Lazy imports for optional deps
_faiss = None
_bm25 = None


def _import_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _import_bm25():
    global _bm25
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        _bm25 = BM25Okapi
    return _bm25


def load_chunks(chunks_path: Path | None = None) -> List[dict]:
    """Load chunk records from chunks.jsonl."""
    path = chunks_path or CHUNKS_JSONL
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_faiss_index(chunks: List[dict]) -> None:
    """Build FAISS index from chunk texts and save index + docstore."""
    ensure_dirs()
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    X = np.array(vectors, dtype=np.float32)
    faiss = _import_faiss()
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (cosine if normalized)
    faiss.normalize_L2(X)
    index.add(X)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    docstore = [{"chunk_id": c["chunk_id"], "source_id": c["source_id"], "text": c["text"]} for c in chunks]
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=0)
    logger.info("Built FAISS index (%d vectors) -> %s, docstore -> %s", len(chunks), FAISS_INDEX_PATH, DOCSTORE_PATH)


def build_bm25_index(chunks: List[dict]) -> None:
    """Build BM25 index (tokenized corpus) and save to BM25_DIR."""
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    tokenized = [c["text"].lower().split() for c in chunks]
    BM25Okapi = _import_bm25()
    bm25 = BM25Okapi(tokenized)
    # Persist: save chunk_ids and tokenized corpus for later
    meta = {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "source_ids": [c["source_id"] for c in chunks],
    }
    with open(BM25_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    # Save tokenized corpus for BM25 (we need to recreate BM25Okapi at load time)
    with open(BM25_DIR / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(tokenized, f, ensure_ascii=False)
    # Pickle/serialize BM25 is not trivial; we'll rebuild at load from corpus + chunk_ids
    import pickle
    with open(BM25_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    logger.info("Built BM25 index (%d docs) -> %s", len(chunks), BM25_DIR)


def build_all_indexes(chunks_path: Path | None = None) -> None:
    """Load chunks, build FAISS + BM25, save to indexes/."""
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError("No chunks to index. Run ingestion and chunking first.")
    build_faiss_index(chunks)
    build_bm25_index(chunks)
