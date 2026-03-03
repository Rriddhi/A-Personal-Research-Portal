"""Entry point for python -m prp. Commands: run_demo | query <query_text> | eval | eval_full | session | export."""
import sys

# .env is loaded via prp.config (imported by run, evaluate, etc.)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m prp run_demo | query <query_text> | eval | eval_full | session <new|list|show <id>|query <id> <query>> | export --session <id> --format csv|json --out <path>")
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
    elif cmd == "session":
        if len(sys.argv) < 3:
            print("Usage: python -m prp session new | list | show <session_id> | query <session_id> <query_text>")
            sys.exit(1)
        sub = sys.argv[2].lower()
        from .session import session_new_cmd, session_list_cmd, session_show_cmd, session_query_cmd
        if sub == "new":
            session_new_cmd()
        elif sub == "list":
            session_list_cmd()
        elif sub == "show":
            if len(sys.argv) < 4:
                print("Usage: python -m prp session show <session_id>")
                sys.exit(1)
            session_show_cmd(sys.argv[3])
        elif sub == "query":
            if len(sys.argv) < 5:
                print("Usage: python -m prp session query <session_id> <query_text>")
                sys.exit(1)
            session_query_cmd(sys.argv[3], " ".join(sys.argv[4:]))
        else:
            print("Unknown session subcommand. Use: new | list | show <id> | query <id> <query_text>")
            sys.exit(1)
    elif cmd == "export":
        # Parse: export --session <id> --format csv|json --out <path>
        args = sys.argv[2:]
        session_id = None
        fmt = None
        out_path = None
        i = 0
        while i < len(args):
            if args[i] == "--session" and i + 1 < len(args):
                session_id = args[i + 1]
                i += 2
            elif args[i] == "--format" and i + 1 < len(args):
                fmt = args[i + 1]
                i += 2
            elif args[i] == "--out" and i + 1 < len(args):
                out_path = args[i + 1]
                i += 2
            else:
                i += 1
        if not session_id or not fmt or not out_path:
            print("Usage: python -m prp export --session <session_id> --format csv|json --out <path>")
            sys.exit(1)
        from .export import export_cmd
        export_cmd(session_id, fmt, out_path)
    else:
        print("Unknown command. Use: run_demo | query | eval | eval_full | session | export")
        sys.exit(1)


if __name__ == "__main__":
    main()
