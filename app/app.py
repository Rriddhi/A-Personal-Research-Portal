"""
Streamlit UI for Personal Research Portal (Phase 3).
Tabs: Search, Ask, Artifacts, History, Evaluation, Method, Data Health.
Run from repo root: streamlit run app/app.py
"""
import sys
from pathlib import Path

import os
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Load .env from repo root so OPENAI_API_KEY is set before any prp code runs
_env_file = REPO / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_env_file)
except ImportError:
    pass
# Fallback: if key still not set, parse .env manually (e.g. when python-dotenv not installed in this env)
if not os.environ.get("OPENAI_API_KEY") and _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip("'\"").strip()
                    if key and val and key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
                        os.environ[key] = val
    except Exception:
        pass

import json
import re
import streamlit as st
import uuid

from prp.config import (
    FAISS_INDEX_PATH,
    DOCSTORE_PATH,
    BM25_DIR,
    OUTPUTS_DIR,
    OUTPUTS_EVAL_DIR,
    EVAL_SUMMARY_MD_OUTPUT,
    AGGREGATE_METRICS_JSON,
    PER_QUERY_METRICS_JSON,
    EVALUATION_SUMMARY_MD,
    TOP_K,
    ensure_dirs,
)
from prp.session import (
    create_session,
    list_sessions,
    get_session,
    append_run,
    list_runs,
    _build_package_for_storage,
)
from prp.quality import validate_package
from prp.manifest_validate import validate_manifest, data_health_stats
from prp.ledger import write_source_selection_ledger
from prp.artifacts import (
    build_evidence_table,
    render_evidence_table_md,
    render_evidence_table_csv,
    build_synthesis_memo,
    save_evidence_table_artifact,
    save_synthesis_memo_artifact,
)
from prp.pdf_export import evidence_table_to_pdf, memo_to_pdf, run_to_pdf
from prp.thread_export import run_to_markdown, session_to_jsonl_bytes, session_to_zip_bytes
from app.components.citation_explorer import render_citation_explorer
from app.components.search_results import render_search_results
try:
    from prp.artifacts import render_evidence_table_html
except ImportError:
    render_evidence_table_html = None
from prp.utils import utc_timestamp

# --- Index check ---
def indexes_ready():
    return (
        FAISS_INDEX_PATH.exists()
        and DOCSTORE_PATH.exists()
        and (BM25_DIR / "bm25.pkl").exists()
    )


def show_index_message():
    st.warning("Indexes not found. Run indexing first: `make ingest` then `make index` (or `make build_index`).")
    if st.button("Show setup commands", key="show_setup_btn"):
        st.code("make setup\nmake ingest\nmake index", language="bash")
    st.stop()


# --- Session state ---
def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "current_run" not in st.session_state:
        st.session_state.current_run = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "_ask_prefill_question" not in st.session_state:
        st.session_state._ask_prefill_question = ""
    if "_search_citation_clipboard" not in st.session_state:
        st.session_state._search_citation_clipboard = []


# --- Search tab ---
def tab_search():
    st.subheader("Search")
    st.caption("Retrieve matching chunks only — no generated answer. Use to explore what’s in the corpus.")
    if not indexes_ready():
        show_index_message()
    query = st.text_input("Query", placeholder="Enter search query", key="search_query")
    top_k = st.slider("Top-k", 1, 20, TOP_K, key="search_top_k")
    method = st.selectbox("Retrieval method", ["RRF (hybrid)", "BM25", "FAISS"], key="search_method")
    if st.button("Search", key="search_btn"):
        if not query.strip():
            st.error("Enter a query")
        else:
            try:
                from prp.retrieve import retrieve_hybrid, retrieve_bm25, retrieve_vector
                if method == "BM25":
                    chunks = retrieve_bm25(query, top_k=top_k)
                elif method == "FAISS":
                    chunks = retrieve_vector(query, top_k=top_k)
                else:
                    chunks = retrieve_hybrid(query, top_k=top_k)
                st.session_state.search_results = {
                    "query": query,
                    "method": method,
                    "chunks": [
                        {
                            "source_id": c.get("source_id", ""),
                            "chunk_id": c.get("chunk_id", ""),
                            "retrieval_method": method,
                            "score": c.get("rrf_score") or c.get("bm25_score") or c.get("vector_score"),
                            "preview": (c.get("text") or "")[:200] + ("..." if len(c.get("text") or "") > 200 else ""),
                            "text": c.get("text") or "",
                        }
                        for c in chunks
                    ],
                }
            except FileNotFoundError as e:
                show_index_message()
            except Exception as e:
                st.error(str(e))
    res = st.session_state.get("search_results")
    if res:
        render_search_results(
            res["chunks"],
            res["query"],
            key_prefix="search",
            method=res.get("method", "semantic similarity"),
        )
        if st.button("Add to session (as run)", key="search_add_session"):
            sid = st.session_state.get("session_id")
            if not sid:
                sid = create_session()
                st.session_state.session_id = sid
            run_id = str(uuid.uuid4())
            run_obj = {
                "run_id": run_id,
                "timestamp": utc_timestamp(),
                "mode": "search",
                "query": res["query"],
                "retrieval_config": {"method": res["method"], "k": top_k},
                "retrieved": [{"source_id": x["source_id"], "chunk_id": x["chunk_id"], "score": x.get("score"), "method": res["method"]} for x in res["chunks"]],
                "retrieved_chunks": [{"source_id": x["source_id"], "chunk_id": x["chunk_id"], "text_preview": x.get("preview")} for x in res["chunks"]],
                "answer": "",
                "diagnostics": {},
                "artifacts": [],
                "ledger_path": "",
            }
            append_run(sid, run_obj)
            st.success(f"Added to session {sid[:8]}...")


# --- Ask tab ---
def tab_ask():
    st.subheader("Ask")
    st.caption("Get a full answer with citations: retrieve chunks then generate a synthesized answer.")
    # Show whether the API key is visible so the user can confirm LLM will be used
    api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    if api_key_set:
        st.caption("OpenAI API key: **set** — answers will use the LLM.")
    else:
        st.warning("OpenAI API key not set. Put `OPENAI_API_KEY=sk-...` in a `.env` file in the project root, or set the env var. Answers will use the fallback (no LLM).")
    if not indexes_ready():
        show_index_message()
    prefill = st.session_state.pop("_ask_prefill_question", "") or ""
    if prefill:
        st.session_state.ask_question = prefill
    question = st.text_area("Question", placeholder="Ask a research question", key="ask_question")
    top_k = st.slider("Top-k", 1, 20, TOP_K, key="ask_top_k")
    method = st.selectbox("Retrieval method", ["RRF (hybrid)", "BM25", "FAISS"], key="ask_method")
    guardrails = st.checkbox("Guardrails (relevance + limitations pass)", value=True, key="ask_guardrails")
    st.caption("Guardrails apply only to RRF (hybrid); BM25/FAISS use single-pass retrieval.")
    if st.button("Run pipeline", key="ask_run_btn"):
        if not question.strip():
            st.error("Enter a question")
        else:
            try:
                from prp.retrieve import retrieve_two_pass, retrieve_hybrid, retrieve_bm25, retrieve_vector
                from prp.generate import generate_answer
                with st.spinner("Retrieving relevant chunks..."):
                    if method == "BM25":
                        chunks = retrieve_bm25(question, top_k=top_k)
                        retrieval_method = "bm25"
                    elif method == "FAISS":
                        chunks = retrieve_vector(question, top_k=top_k)
                        retrieval_method = "faiss"
                    else:
                        if guardrails:
                            chunks = retrieve_two_pass(question, top_k=top_k)
                            retrieval_method = "hybrid_bm25_faiss_rrf_guardrails"
                        else:
                            chunks = retrieve_hybrid(question, top_k=top_k)
                            retrieval_method = "hybrid_bm25_faiss_rrf"
                with st.spinner("Generating answer..."):
                    result = generate_answer(question, chunks)
                result["retrieved_chunks"] = chunks
                result["log_record"] = {
                    "timestamp": utc_timestamp(),
                    "query_id": f"q_{utc_timestamp().replace(':', '-')[:19]}",
                    "query_text": question,
                    "retrieval_method": retrieval_method,
                    "top_k": top_k,
                    "retrieved_chunks": [{"chunk_id": c.get("chunk_id"), "source_id": c.get("source_id"), "text_preview": (c.get("text") or "")[:200]} for c in chunks],
                    "generated_answer": result["answer"],
                    "citations": result.get("citations", []),
                    "citation_mapping": result.get("citation_mapping", []),
                    "model_name": result.get("model_name", ""),
                    "prompt_version": result.get("prompt_version", ""),
                }
                st.session_state.current_run = result
                st.session_state._last_question = question
                st.session_state._top_k = top_k
                st.session_state._ask_method = method
                st.session_state._guardrails = guardrails
            except FileNotFoundError:
                show_index_message()
            except Exception as e:
                st.exception(e)
    run = st.session_state.get("current_run")
    if run:
        model_used = run.get("model_name") or ""
        if model_used and "gpt" in model_used.lower():
            st.caption(f"Model used: **{model_used}** (LLM)")
        else:
            st.caption(f"Model used: **{model_used or 'fallback'}** (no LLM call)")
        # Optional disclaimer when evidence is conflicting (shown above answer, not as the answer)
        if run.get("conflict_disclaimer"):
            st.info(run.get("conflict_disclaimer"))
        answer_text = run.get("answer", "")
        st.markdown(answer_text)
        copy_a, copy_c = st.columns(2)
        with copy_a:
            st.download_button("Copy answer", answer_text, file_name="answer.txt", mime="text/plain", key="copy_answer_btn")
        with copy_c:
            citations_text = "\n".join(
                f"{m.get('apa', '')} → {m.get('chunk_id', '')}"
                for m in run.get("citation_mapping", [])[:30]
            )
            st.download_button("Copy citations", citations_text, file_name="citations.txt", mime="text/plain", key="copy_citations_btn")
        st.caption("Use download to copy to clipboard, or select text above.")
        with st.expander("Diagnostics"):
            n_cit = len(run.get("citations", []))
            n_sent = max(1, len([s for s in (answer_text or "").split(". ") if len(s) > 30]))
            st.metric("Citation coverage (proxy)", f"{min(100, 100 * n_cit / n_sent):.0f}%")
        with st.expander("Evidence / cited chunks (list)", expanded=False):
            for m in run.get("citation_mapping", [])[:15]:
                st.caption(f"{m.get('apa', '')} → {m.get('chunk_id', '')}")
        with st.expander("Citation explorer", expanded=False):
            render_citation_explorer(run)
        if st.button("Save run to session", key="ask_save_session"):
            sid = st.session_state.get("session_id")
            if not sid:
                sid = create_session()
                st.session_state.session_id = sid
            pkg = _build_package_for_storage(run, run.get("log_record", {}))
            pkg["run_id"] = str(uuid.uuid4())
            pkg["mode"] = "ask"
            pkg["query"] = st.session_state.get("_last_question", "")
            pkg["retrieval_config"] = {
                "method": st.session_state.get("_ask_method", "RRF (hybrid)"),
                "k": st.session_state.get("_top_k", TOP_K),
                "guardrails_enabled": st.session_state.get("_guardrails", True),
            }
            valid, errs = validate_package(pkg)
            if valid:
                append_run(sid, pkg)
                st.success(f"Saved to session {sid[:8]}...")
            else:
                st.error("Validation failed: " + ", ".join(errs))


# --- Artifacts tab ---
def tab_artifacts():
    st.subheader("Artifacts")
    run = st.session_state.get("current_run")
    if not run:
        st.info("Run an Ask first, or select a run from History.")
        return
    answer = run.get("answer", "")
    citations = run.get("citations", [])
    chunks = run.get("retrieved_chunks", [])
    if not chunks:
        st.warning("No retrieved chunks for this run.")
        return
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Evidence Table", key="art_evidence_btn"):
            rows = build_evidence_table(answer, citations, chunks)
            st.session_state._evidence_rows = rows
            st.success(f"Built {len(rows)} rows")
    with col2:
        if st.button("Generate Synthesis Memo", key="art_memo_btn"):
            memo = build_synthesis_memo(answer, citations, chunks)
            st.session_state._synthesis_memo = memo
            st.success("Memo generated")
    rows = st.session_state.get("_evidence_rows")
    if rows:
        st.markdown("### Evidence table preview")
        # HTML table so it renders as a real table with wrapping (markdown inside div was shown as raw text)
        st.markdown("""
        <style>
        .evidence-table-wrap table { font-size: 1rem !important; table-layout: fixed; width: 100%; border-collapse: collapse; }
        .evidence-table-wrap td, .evidence-table-wrap th { padding: 0.5rem 0.6rem !important; line-height: 1.4 !important;
          word-wrap: break-word; overflow-wrap: break-word; white-space: normal !important; border: 1px solid #444; }
        .evidence-table-wrap th { text-align: left; }
        .evidence-markdown-export table { font-size: 1rem !important; }
        .evidence-markdown-export td, .evidence-markdown-export th { padding: 0.5rem 0.75rem !important; line-height: 1.4 !important; }
        </style>
        """, unsafe_allow_html=True)
        # Single source of truth: same rows and renderers for preview and all downloads
        if render_evidence_table_html:
            html_table = render_evidence_table_html(rows)
            st.markdown(f'<div class="evidence-table-wrap">\n{html_table}\n</div>', unsafe_allow_html=True)
        else:
            st.markdown(render_evidence_table_md(rows))
        with st.expander("View as dataframe", expanded=False):
            import pandas as pd
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.copy()
                df.insert(0, "#", range(1, len(df) + 1))
                st.dataframe(df[["#", "Claim", "Evidence snippet", "Citation", "Confidence", "Notes"]], use_container_width=True, hide_index=True)
        md_content = render_evidence_table_md(rows)
        csv_content = render_evidence_table_csv(rows)
        with st.expander("View Markdown (for copy/export)", expanded=False):
            st.markdown(md_content)
        st.caption("Download evidence table:")
        ew1, ew2, ew3 = st.columns(3)
        with ew1:
            st.download_button("Download MD", md_content, file_name="evidence_table.md", mime="text/markdown", key="dl_evidence_md")
        with ew2:
            st.download_button("Download CSV", csv_content, file_name="evidence_table.csv", mime="text/csv", key="dl_evidence_csv")
        with ew3:
            try:
                st.download_button("Download PDF", evidence_table_to_pdf(rows, title="Evidence Table"), file_name="evidence_table.pdf", mime="application/pdf", key="dl_evidence_pdf_standalone")
            except Exception as e:
                st.caption(f"PDF: {e}")
        fmt = st.multiselect("Export format", ["md", "csv"], default=["md", "csv"], key="art_export_fmt")
        if st.button("Export and save", key="art_export_btn"):
            sid = st.session_state.get("session_id") or create_session()
            if not st.session_state.get("session_id"):
                st.session_state.session_id = sid
            run_id = (run.get("log_record", {}).get("query_id") or str(uuid.uuid4())).replace(":", "-").replace(" ", "_")
            arts = save_evidence_table_artifact(sid, run_id, rows, fmt)
            st.success(f"Saved to outputs/artifacts/{sid[:8]}.../{run_id}/")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button("Download MD", md_content, file_name="evidence_table.md", mime="text/markdown", key="dl_md")
            with dl_col2:
                st.download_button("Download CSV", csv_content, file_name="evidence_table.csv", mime="text/csv", key="dl_csv")
            with dl_col3:
                try:
                    pdf_bytes = evidence_table_to_pdf(rows, title="Evidence Table", subtitle=run_id)
                    st.download_button("Download PDF", pdf_bytes, file_name="evidence_table.pdf", mime="application/pdf", key="dl_evidence_pdf")
                except Exception as e:
                    st.caption(f"PDF: {e}")
    memo = st.session_state.get("_synthesis_memo")
    if memo:
        st.markdown("### Synthesis memo preview")
        st.markdown(memo[:5000] + ("..." if len(memo) > 5000 else ""))
        try:
            memo_pdf = memo_to_pdf(memo, title="Synthesis Memo", metadata={"run_id": (run.get("log_record", {}).get("query_id") or "")[:80]})
            st.download_button("Download memo PDF", memo_pdf, file_name="synthesis_memo.pdf", mime="application/pdf", key="dl_memo_pdf")
        except Exception as e:
            st.caption(f"Memo PDF: {e}")


# --- History tab ---
def tab_history():
    st.subheader("History")
    ensure_dirs()
    sessions = list_sessions()
    if not sessions:
        if st.button("Create session", key="hist_create_session"):
            sid = create_session()
            st.session_state.session_id = sid
            st.rerun()
        return
    sid_opt = [f"{s['session_id'][:8]}... ({s['num_queries']} runs)" for s in sessions]
    idx = st.selectbox("Session", range(len(sid_opt)), format_func=lambda i: sid_opt[i], key="hist_session")
    if idx is not None:
        sid = sessions[idx]["session_id"]
        st.session_state.session_id = sid
        runs = list_runs(sid)
        st.write(f"**{len(runs)}** runs")
        # Session-level downloads
        st.caption("Session export:")
        sz1, sz2 = st.columns(2)
        with sz1:
            try:
                jsonl_bytes = session_to_jsonl_bytes(sid)
                if jsonl_bytes:
                    st.download_button("Download Session JSONL", jsonl_bytes, file_name=f"session_{sid[:8]}.jsonl", mime="application/x-ndjson", key="dl_session_jsonl")
            except Exception as e:
                st.caption(f"JSONL: {e}")
        with sz2:
            try:
                zip_bytes = session_to_zip_bytes(sid, runs)
                if zip_bytes:
                    st.download_button("Download Session ZIP", zip_bytes, file_name=f"session_{sid[:8]}.zip", mime="application/zip", key="dl_session_zip")
            except Exception as e:
                st.caption(f"ZIP: {e}")
        for i, r in enumerate(runs):
            with st.expander(f"Run {i+1}: {r.get('mode', 'ask')} | {str(r.get('timestamp', ''))[:19]}"):
                st.caption(r.get("query_text", r.get("query", ""))[:150])
                if r.get("answer"):
                    st.text(r["answer"][:300] + "...")
                if r.get("ledger_path"):
                    st.caption(f"Ledger: {r['ledger_path']}")
                if r.get("artifacts"):
                    for a in r["artifacts"]:
                        st.caption(f"Artifact: {a.get('path', '')}")
                # Per-run downloads
                run_id_safe = (r.get("run_id") or f"run_{i}").replace(":", "-").replace(" ", "_")
                rd1, rd2, rd3 = st.columns(3)
                with rd1:
                    st.download_button("Download Run JSON", json.dumps(r, indent=2, ensure_ascii=False), file_name=f"{run_id_safe}.json", mime="application/json", key=f"dl_run_json_{i}")
                with rd2:
                    st.download_button("Download Run Markdown", run_to_markdown(r), file_name=f"{run_id_safe}.md", mime="text/markdown", key=f"dl_run_md_{i}")
                with rd3:
                    try:
                        run_pdf = run_to_pdf(r, title=f"Run {run_id_safe}")
                        st.download_button("Download Run PDF", run_pdf, file_name=f"{run_id_safe}.pdf", mime="application/pdf", key=f"dl_run_pdf_{i}")
                    except Exception as e:
                        st.caption(f"PDF: {e}")
                st.markdown("**Citation explorer**")
                render_citation_explorer(r, key_suffix=f"hist_{i}")


# --- Evaluation tab ---
def tab_evaluation():
    st.caption("Runs on the full query set in `data/eval_queries.json`. Results go to metrics/ or outputs/eval/.")
    with st.expander("What’s the difference between Baseline and Ablations?"):
        st.markdown("""
- **Baseline** = one evaluation run (default retrieval: RRF + guardrails). You get **one set of averages** (e.g. avg_faithfulness, avg_citation_precision) over all eval queries. Shown under \"Baseline\" below.
- **Ablations** = the same queries run with **four retrieval methods** (bm25, faiss, rrf, rrf_guardrails). You get **one set of averages per method** so you can compare. Shown under \"Ablations\" as a comparison table.

\"From last Run eval\" / \"from last Run ablations\" means: the numbers below are from the **last time** you clicked that button. They don’t update until you run again.
        """.strip())
    OUTPUTS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline** — one run with default retrieval (hybrid RRF) and generation.")
        if st.button("Run eval (baseline)", key="eval_baseline_btn"):
            with st.spinner("Running evaluation..."):
                try:
                    from prp.evaluate import run_evaluation, write_phase2_metrics, write_evaluation_summary_md, load_eval_queries
                    agg, per_query = run_evaluation()
                    write_phase2_metrics(per_query, agg)
                    qmap = {q["query_id"]: q for q in load_eval_queries()}
                    failures = [{"query_id": r["query_id"], "query_text": qmap.get(r["query_id"], {}).get("query_text", ""), "issue": "Low scores"} for r in sorted(per_query, key=lambda x: x.get("faithfulness", 0))[:3]]
                    write_evaluation_summary_md(agg, None, failures)
                    st.success("Eval complete. See metrics/")
                except FileNotFoundError:
                    show_index_message()
                except Exception as e:
                    st.exception(e)
    with col2:
        st.markdown("**Ablations** — same queries, 4 retrieval settings (BM25, FAISS, RRF, RRF+guardrails).")
        if st.button("Run ablations (BM25 / FAISS / RRF / RRF+Guardrails)", key="eval_ablate_btn"):
            with st.spinner("Running ablations..."):
                try:
                    from prp.eval_ablate import run_ablations
                    run_ablations()
                    from prp.eval_view import compile_eval_summary_md
                    compile_eval_summary_md()
                    st.success("Ablations complete. See outputs/eval/")
                except FileNotFoundError:
                    show_index_message()
                except Exception as e:
                    st.exception(e)
    st.markdown("---")
    # Baseline: one-line summary + optional download (no full metric dump)
    if AGGREGATE_METRICS_JSON.exists():
        import json
        with open(AGGREGATE_METRICS_JSON, "r", encoding="utf-8") as f:
            baseline_agg = json.load(f)
        faith = baseline_agg.get("avg_faithfulness", "")
        cit = baseline_agg.get("avg_citation_precision", "")
        ctx = baseline_agg.get("avg_context_precision", "")
        st.caption("**Baseline** (last Run eval): faithfulness " + str(faith) + " · citation_precision " + str(cit) + " · context_precision " + str(ctx) + ".")
        bl_col1, bl_col2 = st.columns(2)
        with bl_col1:
            if EVALUATION_SUMMARY_MD.exists():
                st.download_button("Download baseline report (MD)", EVALUATION_SUMMARY_MD.read_text(encoding="utf-8"), file_name="evaluation_summary.md", mime="text/markdown", key="dl_baseline_md")
        with bl_col2:
            if PER_QUERY_METRICS_JSON.exists():
                st.download_button("Download per-query metrics (JSON)", PER_QUERY_METRICS_JSON.read_text(encoding="utf-8"), file_name="per_query_metrics.json", mime="application/json", key="dl_baseline_perquery")
        st.caption("Also in metrics/: aggregate_metrics.json, per_query_metrics.json, evaluation_summary.md")
        # Sortable per-query table + Open as run
        try:
            with open(PER_QUERY_METRICS_JSON, "r", encoding="utf-8") as f:
                per_query = json.load(f)
            if not isinstance(per_query, list):
                per_query = []
        except Exception:
            per_query = []
        try:
            from prp.evaluate import load_eval_queries
            qlist = load_eval_queries()
            qmap = {q["query_id"]: q.get("query_text", "") for q in qlist}
        except Exception:
            qmap = {}
        eval_rows = []
        for r in per_query:
            qid = r.get("query_id", "")
            full_qt = qmap.get(qid, "")
            eval_rows.append({
                "query_id": qid,
                "query_text": full_qt[:80] + ("..." if len(full_qt) > 80 else ""),
                "query_text_full": full_qt,
                "faithfulness": r.get("faithfulness"),
                "citation_precision": r.get("citation_precision"),
                "relevance": r.get("answer_relevance"),
                "coherence": r.get("coherence"),
            })
        if eval_rows:
            sort_by = st.selectbox("Sort by", ["query_id", "faithfulness", "citation_precision", "relevance", "coherence"], key="eval_sort_by")
            reverse = st.checkbox("Descending", value=True, key="eval_sort_desc")
            eval_rows_sorted = sorted(eval_rows, key=lambda x: (x.get(sort_by) is None, x.get(sort_by)), reverse=reverse)
            import pandas as pd
            df_eval = pd.DataFrame(eval_rows_sorted)
            display_cols = ["query_id", "query_text", "faithfulness", "citation_precision", "relevance", "coherence"]
            st.dataframe(df_eval[[c for c in display_cols if c in df_eval.columns]], use_container_width=True, hide_index=True)
            open_idx = st.selectbox("Open as run", range(len(eval_rows_sorted)), format_func=lambda i: f"{eval_rows_sorted[i].get('query_id', '')}: {(eval_rows_sorted[i].get('query_text') or '')[:50]}...", key="eval_open_run")
            if st.button("Populate Ask tab with this query", key="eval_btn_open_run"):
                if open_idx is not None and eval_rows_sorted[open_idx].get("query_text_full"):
                    st.session_state._ask_prefill_question = eval_rows_sorted[open_idx]["query_text_full"]
                    st.success("Go to the **Ask** tab to run this query.")
                else:
                    st.warning("No query text for this row.")
    # Ablations: show comparison table (from new format) or build one from old-format file
    if EVAL_SUMMARY_MD_OUTPUT.exists():
        summary_text = EVAL_SUMMARY_MD_OUTPUT.read_text(encoding="utf-8")
        table_lines = []
        for line in summary_text.splitlines():
            if line.strip().startswith("|"):
                table_lines.append(line)
            elif table_lines:
                break
        if table_lines:
            st.caption("**Ablations** (last Run ablations): comparison across bm25, faiss, rrf, rrf_guardrails.")
            st.markdown("\n".join(table_lines))
        else:
            # Old format: parse ### method and key metrics, show a compact table
            import re
            methods = []
            current = None
            for line in summary_text.splitlines():
                m = re.match(r"^###\s+(bm25|faiss|rrf|rrf_guardrails)\s*$", line.strip())
                if m:
                    current = {"method": m.group(1), "coverage": "—", "unsupported": "—"}
                    methods.append(current)
                elif current is not None and line.strip().startswith("- "):
                    if "avg_citation_coverage:" in line:
                        current["coverage"] = line.split(":", 1)[1].strip().split("[")[0].strip()
                    elif "avg_unsupported_claim_rate:" in line:
                        current["unsupported"] = line.split(":", 1)[1].strip().split("[")[0].strip()
                elif line.strip().startswith("## ") and current is not None:
                    break
            if methods:
                st.caption("**Ablations** (last Run ablations): citation coverage ↑ better, unsupported claim rate ↓ better.")
                tbl = "| Method | citation_coverage | unsupported_claim_rate |\n|--------|-------------------|------------------------|\n"
                tbl += "\n".join(f"| {r['method']} | {r['coverage']} | {r['unsupported']} |" for r in methods)
                st.markdown(tbl)
            else:
                st.caption("**Ablations** (last Run ablations). Re-run ablations to refresh.")
        st.download_button("Download ablation report", EVAL_SUMMARY_MD_OUTPUT.read_text(encoding="utf-8"), file_name="eval_summary.md", mime="text/markdown", key="dl_eval_md")
        st.caption("Also in outputs/eval/: eval_summary.md, bm25/faiss/rrf/rrf_guardrails_metrics.json, *_runs.jsonl (per-query per method)")
    if not AGGREGATE_METRICS_JSON.exists() and not EVAL_SUMMARY_MD_OUTPUT.exists():
        st.info("Run eval (baseline) or Run ablations to see results here.")


# --- Method tab (ledger) ---
def tab_method():
    st.subheader("Method / Source selection ledger")
    runs = []
    sid = st.session_state.get("session_id")
    if sid:
        runs = list_runs(sid)
    if not runs:
        st.info("Select a session and run in History, or run Ask and save to session.")
        return
    idx = st.selectbox("Run", range(len(runs)), format_func=lambda i: runs[i].get("run_id", str(i))[:12], key="method_run")
    if idx is not None:
        r = runs[idx]
        run_id_raw = r.get("run_id") or "run"
        run_id_safe = str(run_id_raw).replace(":", "-").replace(" ", "_")
        # Resolve ledger path: from run record, or session_state (just generated), or default file location
        ledger_path = r.get("ledger_path")
        if ledger_path and Path(ledger_path).exists():
            st.markdown(Path(ledger_path).read_text(encoding="utf-8"))
            return
        default_ledger = OUTPUTS_DIR / "ledgers" / sid / f"{run_id_safe}_source_selection.md"
        if default_ledger.exists():
            st.markdown(default_ledger.read_text(encoding="utf-8"))
            return
        # Show just-generated ledger from this run (same tab session)
        if st.session_state.get("_method_ledger_run_id") == run_id_raw and st.session_state.get("_method_ledger_content"):
            st.success("Ledger generated for this run.")
            st.markdown(st.session_state["_method_ledger_content"])
            return
        st.caption("No ledger for this run. Click below to generate one.")
        if st.button("Generate ledger for this run", key="method_gen_ledger"):
            try:
                stats = data_health_stats()
                # Ensure run has retrieval_config and retrieved for ledger
                run_for_ledger = dict(r)
                if not run_for_ledger.get("retrieval_config"):
                    run_for_ledger["retrieval_config"] = {"method": r.get("retrieval_method", "hybrid_bm25_faiss_rrf"), "k": r.get("top_k", 5), "guardrails_enabled": True}
                if not run_for_ledger.get("retrieved") and run_for_ledger.get("retrieved_chunks"):
                    run_for_ledger["retrieved"] = [{"source_id": c.get("source_id"), "chunk_id": c.get("chunk_id"), "score": c.get("score"), "text_preview": c.get("text_preview")} for c in run_for_ledger["retrieved_chunks"]]
                path = write_source_selection_ledger(run_for_ledger, stats, sid, run_id_safe, "scripted")
                content = Path(path).read_text(encoding="utf-8")
                st.session_state["_method_ledger_run_id"] = run_id_raw
                st.session_state["_method_ledger_content"] = content
                st.success(f"Ledger written: {path}")
                st.markdown(content)
            except Exception as e:
                st.error(f"Failed to generate ledger: {e}")
                import traceback
                st.code(traceback.format_exc())


# --- Data Health tab ---
# Fields the app actually uses for citations and labels (title, authors, year in Ask/Artifacts)
_CRITICAL_MANIFEST_FIELDS = ("doc_id", "title", "authors", "year")

def tab_data_health():
    st.subheader("Data Health")
    st.caption("Manifest metadata is used for **citations** (title, authors, year) in Ask and Artifacts. Missing critical fields mean citations may show IDs instead of paper names.")
    try:
        report = validate_manifest()
        entries = report.get("entries", [])
        total = len(entries)
        missing = report.get("missing_per_field", {})
        parse_rate = report.get("parse_success_rate", 0)

        st.metric("Total entries", total)
        st.metric("Parse success rate", f"{parse_rate:.1%}")

        # Verdict: are critical fields present so citations look good?
        critical_missing = sum(missing.get(f, 0) for f in _CRITICAL_MANIFEST_FIELDS if f in missing)
        if total == 0:
            st.info("No manifest data. Run **make ingest** to build the manifest from your PDFs.")
        elif critical_missing == 0:
            st.success("Critical fields (doc_id, title, authors, year) are present — citations in Ask/Artifacts will show paper names.")
        else:
            st.warning(f"Some entries are missing title, authors, or year ({critical_missing} gaps). Citations may fall back to source IDs. You can edit `data/manifest.csv` to add them.")

        with st.expander("Missing per field (detail)"):
            st.caption("Fields used by the app for citations: doc_id, title, authors, year. Others (e.g. license) are schema-only and do not affect Search/Ask.")
            st.json(missing)

        unknown = report.get("unknown_fields", [])
        if unknown:
            with st.expander("Unknown fields"):
                st.caption("Columns in the manifest that are not in the schema. Safe to ignore unless you need them for export.")
                st.write(unknown)
    except Exception as e:
        st.error(str(e))


# --- Main ---
def main():
    st.set_page_config(page_title="Personal Research Portal", layout="wide")
    init_state()
    st.title("Personal Research Portal")
    ensure_dirs()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Search", "Ask", "Artifacts", "History", "Evaluation", "Method", "Data Health",
    ])
    with tab1:
        tab_search()
    with tab2:
        tab_ask()
    with tab3:
        tab_artifacts()
    with tab4:
        tab_history()
    with tab5:
        tab_evaluation()
    with tab6:
        tab_method()
    with tab7:
        tab_data_health()


if __name__ == "__main__":
    main()
