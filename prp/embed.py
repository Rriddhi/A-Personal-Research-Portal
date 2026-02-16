"""Embeddings: sentence-transformers (local, no API)."""
from pathlib import Path
from typing import List

from .config import EMBEDDING_MODEL
from .utils import logger

_cached_model = None


def get_embedding_model():
    global _cached_model
    if _cached_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _cached_model = SentenceTransformer(EMBEDDING_MODEL)
    return _cached_model


def embed_texts(texts: List[str]) -> "List[List[float]]":
    """Return list of embedding vectors for each text."""
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=len(texts) > 50).tolist()
