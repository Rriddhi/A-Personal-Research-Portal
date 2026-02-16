"""Entry point for python -m prp. Commands: run_demo | query <query_text> | eval | eval_full."""
import sys

# .env is loaded via prp.config (imported by run, evaluate, etc.)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m prp run_demo | query <query_text> | eval | eval_full")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "run_demo":
        from .run import run_demo
        run_demo()
    elif cmd == "query":
        from .run import run_query
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What evidence exists for personalized health recommendations?"
        r = run_query(q)
        print(r["answer"])
    elif cmd == "eval":
        from prp.evaluate import run_evaluation, write_phase2_metrics, write_evaluation_summary_md, load_eval_queries
        agg, per_query = run_evaluation()
        write_phase2_metrics(per_query, agg)
        qmap = {q["query_id"]: q for q in load_eval_queries()}
        failures = [{"query_id": r["query_id"], "query_text": qmap.get(r["query_id"], {}).get("query_text", ""), "issue": "Low scores"} for r in sorted(per_query, key=lambda x: x.get("faithfulness", 0))[:3]]
        write_evaluation_summary_md(agg, None, failures)
        print("Evaluation complete. See metrics/")
    elif cmd == "eval_full":
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        script = repo / "scripts" / "run_evaluation_full.py"
        import subprocess
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(repo))
    else:
        print("Unknown command. Use: run_demo | query | eval | eval_full")
        sys.exit(1)


if __name__ == "__main__":
    main()
