#!/usr/bin/env python3
"""Quick test: load .env and call generate_answer with LLM (no retrieval). Run from repo root."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Load .env like the app
_env = REPO / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        for line in open(_env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k in ("OPENAI_API_KEY", "OPENAI_MODEL"):
                    os.environ[k] = v

key_set = bool(os.environ.get("OPENAI_API_KEY"))
print("OPENAI_API_KEY set:", key_set)
if not key_set:
    print("Add OPENAI_API_KEY to .env in repo root and re-run.")
    sys.exit(1)

# Call the real generation path (no chunks => LLM-only answer)
from prp.generate import generate_answer

result = generate_answer("What is 2 + 2? Answer in one short sentence.", [], use_llm=True)
model = result.get("model_name", "")
answer = (result.get("answer", "") or "").strip()
print("Model used:", model)
print("Answer preview:", (answer[:200] + "..." if len(answer) > 200 else answer) or "(empty)")
if model and "gpt" in model.lower() and answer:
    print("OK: LLM generation is working.")
else:
    print("Unexpected: expected model name containing 'gpt' and non-empty answer.")
    sys.exit(1)
