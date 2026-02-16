"""Configuration and paths for Phase 2 RAG pipeline."""
import os
from pathlib import Path

# Load .env from repo root if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Repo root: directory containing data, prp (optional: Research_Papers for --source)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Optional: alternate ingest source (run_ingest.py --source Research_Papers)
RESEARCH_PAPERS_DIR = REPO_ROOT / "Research_Papers"

# Data paths
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SOURCES_JSONL = PROCESSED_DIR / "sources.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
EVAL_QUERIES_JSON = DATA_DIR / "eval_queries.json"

# Indexes
INDEXES_DIR = REPO_ROOT / "indexes"
FAISS_INDEX_PATH = INDEXES_DIR / "faiss.index"
DOCSTORE_PATH = INDEXES_DIR / "docstore.json"
BM25_DIR = INDEXES_DIR / "bm25"

# Logs and metrics
LOGS_DIR = REPO_ROOT / "logs"
RUNS_DIR = LOGS_DIR / "runs"
PHASE2_RUNS_JSONL = RUNS_DIR / "phase2_runs.jsonl"
METRICS_DIR = REPO_ROOT / "metrics"
EVAL_SUMMARY_JSON = METRICS_DIR / "phase2_eval_summary.json"
EVAL_PER_QUERY_CSV = METRICS_DIR / "phase2_eval_per_query.csv"
PER_QUERY_METRICS_JSON = METRICS_DIR / "per_query_metrics.json"
AGGREGATE_METRICS_JSON = METRICS_DIR / "aggregate_metrics.json"
EVALUATION_SUMMARY_MD = METRICS_DIR / "evaluation_summary.md"

# Chunking defaults
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5
RRF_K = 60  # reciprocal rank fusion constant
MAX_PER_SOURCE = 2  # cap chunks per source_id in top_k for diversification (fill remaining from next best)

# Post-retrieval relevance: re-rank by RRF + embedding similarity; drop below min similarity
RELEVANCE_RRF_WEIGHT = 0.3  # weight for RRF score in combined re-rank
RELEVANCE_SIM_WEIGHT = 0.7  # weight for query–chunk cosine similarity
RELEVANCE_MIN_SIMILARITY = 0.25  # drop chunks with cos_sim below this (calibrated on effectiveness+limitations, general queries)
# Topic mismatch guardrail: prevents domain drift; improves relevance when query doesn't ask about oncology
TOPIC_MISMATCH_GUARDRAIL = True
ONCOLOGY_TERMS = ("cancer", "chemotherapy", "tumor", "oncology", "radiotherapy", "metastasis")
# Penalty when chunk contains oncology terms but query does not (soft fallback if filtering would drop below top_k)
TOPIC_MISMATCH_PENALTY = 0.05  # strong down-rank; filter preferred when enough non-oncology chunks exist

# Bibliography filter: chunks with score >= threshold are excluded from retrieval
BIBLIOGRAPHY_THRESHOLD = 0.5  # strict: filter reference list fragments
BIBLIOGRAPHY_THRESHOLD_RELAXED = 0.7  # fallback: if < top_k after filter, retry with this (fewer filtered)

# Embedding model (local, no API key)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Generation
PROMPT_VERSION = "v1"
MODEL_NAME = "local_extractive"  # or set via env for LLM
MIN_SUPPORT_BULLETS = 5
MIN_LIMITATIONS_BULLETS = 3
# Claim-evidence alignment: min fraction of query keywords that must appear in retrieved chunks
KEYWORD_OVERLAP_THRESHOLD = 0.15

# Phase 2 output structure (Assignment spec)
ANSWER_MIN_WORDS = 250
ANSWER_MAX_WORDS = 500
MIN_EVIDENCE_SNIPPETS = 4

# Ensure directories exist
def ensure_dirs():
    for d in (
        DATA_DIR, RAW_DIR, PROCESSED_DIR,
        INDEXES_DIR, BM25_DIR,
        LOGS_DIR, RUNS_DIR, METRICS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
