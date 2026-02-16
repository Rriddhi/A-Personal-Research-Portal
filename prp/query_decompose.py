"""Lightweight query decomposition for compare/difference queries (no LLM)."""
import re
from typing import List, Optional, Tuple


def is_compare_query(query: str) -> bool:
    """True if query asks for comparison, difference, or agree/disagree."""
    if not query or not isinstance(query, str):
        return False
    q = query.strip().lower()
    if q.startswith("compare"):
        return True
    if "compare" in q or "difference" in q or "differences" in q:
        return True
    if "agree" in q and "disagree" in q:
        return True
    if "agree or disagree" in q or "agree and disagree" in q:
        return True
    if re.search(r"\bvs\.?\b|\bversus\b", q):
        return True
    return False


def extract_compare_entities(query: str) -> Optional[Tuple[str, str]]:
    """
    Extract A and B from compare-style query. Returns (A, B) or None.
    Tries: "A vs B", "A versus B", "compare A and B", "difference(s) between A and B",
    "do A and B agree", "A and B agree or disagree".
    """
    if not query or not isinstance(query, str):
        return None
    q = query.strip()
    if not q:
        return None

    # 1) "A vs B" or "A versus B" (case-insensitive)
    m = re.search(r"^(.+?)\s+vs\.?\s+(.+?)\s*[?.]?\s*$", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)
    m = re.search(r"^(.+?)\s+versus\s+(.+?)\s*[?.]?\s*$", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)

    # 2) "Compare A and B" or "compare A with B"
    m = re.match(r"compare\s+(.+?)\s+and\s+(.+?)\s*[?.]?\s*$", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)
    m = re.match(r"compare\s+(.+?)\s+with\s+(.+?)\s*[?.]?\s*$", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)

    # 3) "difference(s) between A and B"
    m = re.search(r"difference(s)?\s+between\s+(.+?)\s+and\s+(.+?)\s*[?.]?\s*$", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(2).strip(), m.group(3).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)

    # 4) "do A and B agree" / "A and B agree or disagree"
    m = re.search(r"(?:do\s+)?(.+?)\s+and\s+(.+?)\s+(?:agree|disagree)", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)

    # 5) Fallback: any "X vs Y" or "X versus Y" in the middle
    m = re.search(r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)
    m = re.search(r"(.+?)\s+versus\s+(.+?)(?:\?|$)", q, re.IGNORECASE | re.DOTALL)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b and len(a) > 1 and len(b) > 1:
            return (a, b)

    return None


def build_subqueries(query: str) -> Optional[List[str]]:
    """
    If query is compare-style, return 2–3 subqueries:
    ["Definition of A", "Definition of B", "differences between A and B"].
    Otherwise return None.
    """
    if not is_compare_query(query):
        return None
    entities = extract_compare_entities(query)
    if not entities:
        return None
    a, b = entities
    return [
        f"Definition of {a}",
        f"Definition of {b}",
        f"differences between {a} and {b}",
    ]
