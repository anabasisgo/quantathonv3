#!/usr/bin/env python3
"""Preflight a Quantathon harness submission without opening circuit contents.

Inventory deliberately mirrors quantathon-harness/run.py: recurse all paths,
select regular files whose exact suffix is .qasm or .zst, and strip only the
final .zst from a selected file's basename. Expected thresholds are the
challenge's fixed 16, 64, and 512 values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


THRESHOLDS = (16, 64, 512)
CSV_COLUMNS = ["team", "filename", "threshold", "pred_duration_s", "parse_s", "predict_s"]
TIMER_CAP_S = 15.0


class PreflightError(Exception):
    pass


def harness_name(path: Path) -> str:
    """Exactly mirror run.py's `path.name[:-4] if suffix == '.zst'` logic."""
    return path.name[:-4] if path.suffix == ".zst" else path.name


def inventory(circuits_dir: Path) -> dict:
    if not circuits_dir.exists():
        raise PreflightError(f"Circuits directory does not exist: {circuits_dir}")
    if not circuits_dir.is_dir():
        raise PreflightError(f"Circuits path is not a directory: {circuits_dir}")
    # Keep suffix handling case-sensitive and include every .zst, as the
    # harness does; do not attempt to inspect/decompress any file content.
    files = sorted(p for p in circuits_dir.rglob("*")
                   if p.suffix in (".qasm", ".zst") and p.is_file())
    if not files:
        raise PreflightError(f"No .qasm / .qasm.zst files found under {circuits_dir}/")

    names: dict[str, list[str]] = {}
    selected = []
    suspicious_zst = []
    for path in files:
        normalized = harness_name(path)
        rel = path.relative_to(circuits_dir).as_posix()
        names.setdefault(normalized, []).append(rel)
        selected.append({"path": rel, "filename": normalized, "suffix": path.suffix})
        if path.suffix == ".zst" and not path.name.endswith(".qasm.zst"):
            suspicious_zst.append(rel)

    collisions = [
        {"filename": name, "paths": paths}
        for name, paths in sorted(names.items()) if len(paths) > 1
    ]
    expected = [
        {"filename": name, "threshold": threshold}
        for name in sorted(names) for threshold in THRESHOLDS
    ]
    errors = []
    if collisions:
        details = "; ".join(f"{c['filename']}: {', '.join(c['paths'])}" for c in collisions)
        errors.append("Conflicting circuit keys collapse to the same harness filename: " + details)
    warnings = []
    if suspicious_zst:
        warnings.append("Harness selects .zst files that do not end in .qasm.zst: "
                        + ", ".join(suspicious_zst))
    return {
        "circuits_dir": str(circuits_dir),
        "selected_file_count": len(files),
        "circuits": selected,
        "thresholds": list(THRESHOLDS),
        "expected_row_count": len(expected),
        "expected_keys": expected,
        "filename_collisions": collisions,
        "suspicious_unrelated_zst": suspicious_zst,
        "errors": errors,
        "warnings": warnings,
    }


def _csv_key(row: dict, row_number: int):
    filename = (row.get("filename") or "").strip()
    if not filename:
        raise ValueError(f"row {row_number}: filename is blank")
    threshold_text = (row.get("threshold") or "").strip()
    try:
        threshold_value = float(threshold_text)
        if not math.isfinite(threshold_value):
            raise ValueError
        # Match score.py's key normalization: int(float(threshold)).
        threshold = int(threshold_value)
    except (ValueError, OverflowError):
        raise ValueError(f"row {row_number}: invalid threshold {threshold_text!r}") from None
    return filename, threshold


def validate_csv(csv_path: Path, expected_keys: list[dict]) -> dict:
    expected = {(x["filename"], int(x["threshold"])) for x in expected_keys}
    errors = []
    actual_rows: dict[tuple[str, int], list[int]] = {}
    row_count = 0
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_COLUMNS:
                errors.append(f"CSV columns must be exactly {CSV_COLUMNS!r}; got {reader.fieldnames!r}")
                return {"path": str(csv_path), "row_count": 0, "errors": errors}
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                if None in row:
                    errors.append(f"row {row_number}: extra fields beyond the required columns")
                    continue
                try:
                    key = _csv_key(row, row_number)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                actual_rows.setdefault(key, []).append(row_number)

                for column in ("pred_duration_s", "parse_s", "predict_s"):
                    raw = (row.get(column) or "").strip()
                    try:
                        value = float(raw)
                    except ValueError:
                        errors.append(f"row {row_number}: {column} is not numeric")
                        continue
                    if not math.isfinite(value):
                        errors.append(f"row {row_number}: {column} must be finite")
                    elif column == "pred_duration_s" and value <= 0:
                        errors.append(f"row {row_number}: pred_duration_s must be positive")
                    elif column in ("parse_s", "predict_s") and value < 0:
                        errors.append(f"row {row_number}: {column} must be nonnegative")
                    elif column in ("parse_s", "predict_s") and value > TIMER_CAP_S:
                        errors.append(f"row {row_number}: {column} exceeds the {TIMER_CAP_S:g}s cap")
    except OSError as exc:
        return {"path": str(csv_path), "row_count": row_count,
                "errors": [f"Cannot read CSV: {exc}"]}

    duplicates = {f"{name}@{threshold}": rows for (name, threshold), rows in actual_rows.items()
                  if len(rows) > 1}
    if duplicates:
        errors.append("Duplicate filename/threshold keys: " + ", ".join(duplicates))
    actual = set(actual_rows)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"Missing {len(missing)} expected filename/threshold keys: "
                      + ", ".join(f"{n}@{t}" for n, t in missing[:20]))
    if extra:
        errors.append(f"Found {len(extra)} unexpected filename/threshold keys: "
                      + ", ".join(f"{n}@{t}" for n, t in extra[:20]))
    return {
        "path": str(csv_path), "row_count": row_count,
        "unique_key_count": len(actual), "duplicate_keys": duplicates,
        "missing_key_count": len(missing), "missing_keys": [list(k) for k in missing],
        "extra_key_count": len(extra), "extra_keys": [list(k) for k in extra],
        "errors": errors,
    }


def preflight(circuits_dir: Path, csv_path: Path | None = None) -> dict:
    report = inventory(circuits_dir)
    if csv_path is not None:
        report["csv_validation"] = validate_csv(csv_path, report["expected_keys"])
        report["errors"].extend(report["csv_validation"]["errors"])
    report["status"] = "failed" if report["errors"] else "passed"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", required=True, type=Path,
                        help="Circuit folder scanned recursively using run.py's suffix rules")
    parser.add_argument("--csv", type=Path, help="Optional submission CSV to validate")
    parser.add_argument("--report", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    try:
        report = preflight(args.circuits, args.csv)
    except PreflightError as exc:
        report = {"status": "failed", "circuits_dir": str(args.circuits),
                  "errors": [str(exc)], "warnings": []}

    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        except OSError as exc:
            print(f"Could not write JSON report {args.report}: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
