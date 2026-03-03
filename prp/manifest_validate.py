"""
Manifest schema validation and autofill. Validates manifest (CSV or JSON list),
autofills missing required fields with null + unknown_reason, and produces data health stats.
"""
import csv
import json
from pathlib import Path
from typing import Any

from .config import MANIFEST_PATH

_SCHEMA_PATH = Path(__file__).resolve().parent / "manifest_schema.json"

def _load_schema() -> dict:
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_entry(entry: dict, schema: dict) -> dict:
    """Normalize keys: map source_id -> doc_id etc. for validation."""
    out = dict(entry)
    aliases = schema.get("aliases") or {}
    for canonical, alts in aliases.items():
        if canonical in out:
            continue
        for alt in alts:
            if alt in out and out[alt] not in (None, ""):
                out[canonical] = out[alt]
                break
        else:
            if "source_id" in out and canonical == "doc_id":
                out["doc_id"] = out.get("source_id")
            if "local_path" in out and canonical == "source_path":
                out["source_path"] = out.get("local_path")
            if "type" in out and canonical == "source_type":
                out["source_type"] = out.get("type")
    return out


def autofill_missing_fields(entry: dict, schema: dict) -> dict:
    """Fill missing required fields with null and set unknown_reason. Return new dict."""
    required = set(schema.get("required", []))
    optional = set(schema.get("optional", []))
    all_known = required | optional
    entry = _normalize_entry(entry, schema)
    out = dict(entry)
    missing = []
    for k in required:
        val = out.get(k)
        if val is None or (isinstance(val, str) and not str(val).strip()):
            out[k] = None
            missing.append(k)
    unknown_keys = [k for k in out if k not in all_known and k not in ("unknown_reason",)]
    if missing or unknown_keys:
        reasons = []
        if missing:
            reasons.append(f"missing:{','.join(missing)}")
        if unknown_keys:
            reasons.append(f"unknown_fields:{','.join(unknown_keys)}")
        out["unknown_reason"] = "; ".join(reasons)
    return out


def validate_manifest(manifest_path: Path | None = None) -> dict:
    """
    Validate manifest (CSV or JSON). Returns report dict:
    - valid: bool
    - entries: list of (normalized) entries
    - missing_per_field: {field: count}
    - unknown_fields: list of field names seen that are not in schema
    - parse_errors: list of str
    - parse_success_rate: float (if applicable)
    """
    path = manifest_path or MANIFEST_PATH
    path = Path(path)
    schema = _load_schema()
    required = set(schema.get("required", []))
    optional = set(schema.get("optional", []))
    all_known = required | optional

    report: dict = {
        "valid": True,
        "entries": [],
        "missing_per_field": {k: 0 for k in required},
        "unknown_fields": [],
        "parse_errors": [],
        "parse_success_rate": 1.0,
    }
    seen_keys: set = set()
    entries: list = []

    if not path.exists():
        report["parse_errors"].append(f"Manifest not found: {path}")
        report["valid"] = False
        return report

    if path.suffix.lower() == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_entries = data if isinstance(data, list) else data.get("entries", data.get("sources", []))
        except Exception as e:
            report["parse_errors"].append(str(e))
            report["valid"] = False
            return report
    else:
        raw_entries = []
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_entries.append(dict(row))
        except Exception as e:
            report["parse_errors"].append(str(e))
            report["valid"] = False
            return report

    for i, raw in enumerate(raw_entries):
        entry = _normalize_entry(raw, schema)
        for k in entry:
            seen_keys.add(k)
        for k in required:
            val = entry.get(k)
            if val is None or (isinstance(val, str) and not str(val).strip()):
                report["missing_per_field"][k] = report["missing_per_field"].get(k, 0) + 1
        filled = autofill_missing_fields(raw, schema)
        entries.append(filled)

    report["entries"] = entries
    report["unknown_fields"] = sorted(seen_keys - all_known - {"unknown_reason"})
    if raw_entries:
        report["parse_success_rate"] = 1.0 - (sum(report["missing_per_field"].values()) / (len(raw_entries) * max(1, len(required))))
    report["valid"] = len(report["parse_errors"]) == 0
    return report


def write_clean_manifest(
    manifest_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Validate, autofill, and write cleaned manifest (CSV). Returns output path."""
    path = manifest_path or MANIFEST_PATH
    out_path = Path(out_path) if out_path else path.parent / "manifest.cleaned.csv"
    report = validate_manifest(path)
    entries = report.get("entries", [])
    if not entries:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write("doc_id,title,authors,year,source_path,source_type,license,unknown_reason\n")
        return out_path
    all_keys = set()
    for e in entries:
        all_keys.update(e.keys())
    fieldnames = sorted(all_keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)
    return out_path


def data_health_stats(manifest_path: Path | None = None) -> dict:
    """Produce data health stats: missing per field, parse success rate, chunk counts if available."""
    report = validate_manifest(manifest_path)
    total = len(report.get("entries", []))
    missing = report.get("missing_per_field", {})
    parse_rate = report.get("parse_success_rate", 1.0)
    return {
        "total_entries": total,
        "missing_per_field": missing,
        "parse_success_rate": parse_rate,
        "unknown_fields": report.get("unknown_fields", []),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate and clean manifest")
    p.add_argument("--manifest", default=None, help="Manifest CSV or JSON path")
    p.add_argument("--out", default=None, help="Output cleaned path (default: data/manifest.cleaned.csv)")
    args = p.parse_args()
    path = Path(args.manifest) if args.manifest else MANIFEST_PATH
    out = Path(args.out) if args.out else path.parent / "manifest.cleaned.csv"
    write_clean_manifest(path, out)
    print(f"Wrote {out}")
    r = validate_manifest(path)
    print("Missing per field:", r.get("missing_per_field"))
    print("Unknown fields:", r.get("unknown_fields"))


if __name__ == "__main__":
    main()
