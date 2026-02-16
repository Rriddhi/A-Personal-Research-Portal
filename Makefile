# Phase 2 RAG - data/raw-first pipeline
.PHONY: ingest merge_bib_manifest merge_bib_manifest_write build_index phase2_demo run_eval enrich_manifest clean run_query test

# Ingest: discover PDFs from data/raw/; writes data/processed/sources.jsonl and data/manifest.csv
ingest:
	python scripts/run_ingest.py

# Merge Zotero BibTeX into manifest (preview only; writes data/manifest_from_bib.csv)
merge_bib_manifest:
	python scripts/merge_bib_manifest.py

# Merge and overwrite data/manifest.csv with BibTeX metadata
merge_bib_manifest_write:
	python scripts/merge_bib_manifest.py --write

# Chunk + build FAISS + BM25 indexes
build_index:
	python scripts/build_index.py

# One-command demo: build index if needed, run one query, print answer + citations
phase2_demo:
	python -m prp run_demo

# Run evaluation set (20 queries), write metrics/
run_eval:
	python scripts/run_eval.py

enrich_manifest:
	python scripts/enrich_manifest.py

# Remove processed data, indexes, and metrics (keeps data/raw/)
clean:
	rm -rf data/processed indexes metrics

run_query:
	@echo "Usage: python scripts/run_query.py \"your query\" [query_id]"

test:
	python -m pytest tests/test_filters_and_evidence.py -v
