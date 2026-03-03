# Personal Research Portal

A local RAG app over your PDF corpus: **search** chunks, **ask** questions and get cited answers, and build **evidence tables** and memos. Uses hybrid retrieval (BM25 + FAISS) and optional OpenAI for answer synthesis.

---

## Quick start

From the **project root** (folder containing `prp/`, `app/`, `data/`):

**1. Environment**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pymupdf` fails to install, use conda first: `conda install -c conda-forge pymupdf` then `pip install -r requirements.txt`.

**2. Corpus and indexes**

- Put PDFs in **`data/raw/`**
- Then run:

```bash
make ingest    # extract text, chunk, build manifest
make index     # build FAISS + BM25 indexes
```

**3. Run the app**

```bash
make app
```

Open **http://localhost:8501**.

**4. (Optional) LLM answers**

To get synthesized answers with citations instead of the fallback, add a **`.env`** file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

The Ask tab will show “OpenAI API key: set” when the key is loaded.

---

## What you can do in the app

| Tab | Use |
|-----|-----|
| **Search** | Run a query; see matching chunks (no generated answer). Choose RRF (hybrid), BM25, or FAISS. |
| **Ask** | Ask a question; get a synthesized answer with citations (LLM if key set, else fallback). |
| **Artifacts** | From an Ask run: build evidence table and synthesis memo; export MD/CSV. |
| **History** | View sessions and runs; open a run and load it into Artifacts. |
| **Evaluation** | View evaluation summary (after running `make eval`). |
| **Method** | Pipeline description; generate source-selection ledger for a run. |
| **Data Health** | Manifest validation and basic stats. |

---

## Optional: CLI and evaluation

- **Single query:** `python -m prp query "Your question"`
- **Sessions / export:** `python -m prp session new` then `python -m prp session query <id> "..."`; `python -m prp export --session <id> --format csv --out evidence.csv`
- **Evaluation:** `make eval` (writes to `outputs/eval/`)

---

## If something fails

- **`ModuleNotFoundError: No module named 'numpy'` (or similar)**  
  Activate the same environment you used for `pip install`, then run the command again. Or run `pip install -r requirements.txt` in that environment.

- **“OpenAI API key not set” in the app**  
  Ensure `.env` is in the **project root** (same folder as `prp/` and `app/`) and contains exactly `OPENAI_API_KEY=sk-...` with no spaces around `=`.

- **Indexes not found**  
  Run `make ingest` then `make index` from the project root before starting the app.
