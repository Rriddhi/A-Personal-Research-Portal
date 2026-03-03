# Phase 2 RAG - data/raw-first pipeline
.PHONY: setup ingest merge_bib_manifest merge_bib_manifest_write build_index index phase2_demo phase3_demo run_eval eval app eval_ablate enrich_manifest clean run_query test test_phase3

# One-time setup: venv + install deps (run from repo root)
setup:
	python -m venv .venv 2>/dev/null || true
	. .venv/bin/activate 2>/dev/null || true; pip install -r requirements.txt

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

# Alias for build_index
index:
	$(MAKE) build_index

# One-command demo: build index if needed, run one query, print answer + citations
phase2_demo:
	python -m prp run_demo

# Phase 3 demo: create session, run one query in session, export CSV+JSON
phase3_demo:
	@sid=$$(python -m prp session new | tr -d '\n') && \
	python -m prp session query "$$sid" "What evidence exists for personalized health recommendations?" > /dev/null && \
	python -m prp export --session "$$sid" --format csv --out logs/sessions/phase3_export_sample.csv && \
	python -m prp export --session "$$sid" --format json --out logs/sessions/phase3_export_sample.json && \
	echo "Phase 3 demo complete. Session ID: $$sid" && \
	echo "Exports: logs/sessions/phase3_export_sample.csv, logs/sessions/phase3_export_sample.json"

# Run evaluation set (20 queries), write metrics/
run_eval:
	python scripts/run_eval.py

# Alias for run_eval
eval:
	$(MAKE) run_eval

# Streamlit UI (Phase 3)
app:
	streamlit run app/app.py

# Ablation suite: BM25 / FAISS / RRF / RRF+guardrails; writes outputs/eval/
eval_ablate:
	python -m prp.eval_ablate

enrich_manifest:
	python scripts/enrich_manifest.py

# Remove processed data, indexes, and metrics (keeps data/raw/)
clean:
	rm -rf data/processed indexes metrics

run_query:
	@echo "Usage: python scripts/run_query.py \"your query\" [query_id]"

test:
	python -m pytest tests/test_filters_and_evidence.py -v

test_phase3:
	python -m pytest tests/test_phase3.py -v
