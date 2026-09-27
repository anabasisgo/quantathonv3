"""Fit and publish the evaluated 145-input mock-simulation runtime model.

Run from the repository root:
    uv run --offline --cache-dir /tmp/quantathon-uv-cache \
        --with scikit-learn==1.9.1 --with xgboost-cpu==3.4.1 \
        python quantathon-harness/train_mock.py

This preserves the experiment's seed-42 recipe, verifies every live feature
against the recorded experiment, and checks the portable artifact before an
atomic replacement. It does not reselect features, seeds or routing settings.
The earlier cross-validation scores remain separate from this full-data fit.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import sklearn

from model import (CAP_SECONDS, DEFAULT_ARTIFACT_PATH, FEATURE_NAMES,
                   MOCK_FEATURE_NAMES, RuntimeModel, _feature_vector)
from run import read_qasm
from train_improved import (apply_recipe, export_fitted, fit_family, load_data,
                            predict_family)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "reports/mock-simulation"
RECIPE = {"family": "routed_2", "fast_confidence": .85,
          "regular_confidence": .85, "timeout_confidence": .35}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    ap.add_argument("--report", type=Path, default=EXPERIMENT / "implementation")
    args = ap.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((EXPERIMENT / "provenance.json").read_text())
    for relative in ("runtime-data.csv", "reports/model-improvement/enhanced-features.json",
                     "reports/model-improvement/baseline-features.json",
                     "reports/mock-simulation/mock_simulation.py"):
        if digest(ROOT / relative) != provenance["hashes"][relative]:
            raise ValueError(f"Experiment input changed: {relative}; rerun validation before promotion")
    if (list(MOCK_FEATURE_NAMES) != provenance["features"] or RECIPE != provenance["recipe"]
            or sklearn.__version__ != provenance["sklearn"] or np.__version__ != provenance["numpy"]):
        raise ValueError("Feature schema, recipe or dependencies differ from the evaluated experiment")
    if digest(EXPERIMENT / "features.json") != provenance["features_sha256"]:
        raise ValueError("The experiment feature cache has changed")
    reference = load_module("mock_reference", EXPERIMENT / "mock_simulation.py")
    mock_cache = json.loads((EXPERIMENT / "features.json").read_text())["circuits"]
    _, y, groups, timeout, keys, structural_cache = load_data(
        ROOT / "runtime-data.csv", ROOT / "reports/model-improvement")
    parser = RuntimeModel(load_model=False, use_mock=True)
    features, timings = {}, []
    for i, (name, old) in enumerate(sorted(structural_cache.items()), 1):
        cached = mock_cache[name]
        if cached["compressed_sha256"] != old["compressed_sha256"]:
            raise ValueError(f"Mock feature cache refers to another circuit: {name}")
        source = read_qasm(ROOT / "training_circuits" / (name + ".zst"))
        start = time.perf_counter()
        f = parser.featurize(source)
        elapsed = time.perf_counter() - start
        if elapsed >= 15:
            raise RuntimeError(f"Featurizer exceeded 15 seconds: {name}, {elapsed}")
        if any(f[k] != v for k, v in old["features"].items()):
            raise ValueError(f"Structural feature mismatch: {name}")
        expected_mock = {**cached["controls"], **cached["entropy_features"]}
        if any(f[k] != v for k, v in expected_mock.items()):
            raise ValueError(f"Mock feature mismatch: {name}")
        for threshold in (16, 64, 512):
            row = {**old["features"], **expected_mock,
                   **reference.threshold_features(cached, threshold),
                   "threshold": threshold, "log2_threshold": math.log2(threshold)}
            expected = np.array([row[k] for k in MOCK_FEATURE_NAMES], dtype=np.float32)
            actual = np.array(_feature_vector(f, threshold, MOCK_FEATURE_NAMES), dtype=np.float32)
            if not np.array_equal(actual, expected):
                raise ValueError(f"145-input vector mismatch: {name}, {threshold}")
        features[name] = f
        timings.append({"filename": name, "featurize_seconds": elapsed,
                        "mock_coverage": f["mock_coverage"], "mock_incomplete": f["mock_incomplete"]})
        if i % 50 == 0 or i == len(structural_cache):
            print(f"Verified {i}/{len(structural_cache)} circuits; slowest "
                  f"{max(r['featurize_seconds'] for r in timings):.3f}s", flush=True)

    x = np.asarray([_feature_vector(features[name], threshold, MOCK_FEATURE_NAMES)
                    for name, threshold in keys], dtype=np.float32)
    fitted = fit_family("routed_2", x, y, groups, timeout)
    component = export_fitted(fitted, RECIPE, x)
    native = apply_recipe(predict_family(fitted, x), RECIPE)
    summary = json.loads((EXPERIMENT / "summary.json").read_text())
    seed_results = json.loads((EXPERIMENT / "seed-results.json").read_text())
    metadata = {"model": "routed_2_mock_bond_cost", "recipe": RECIPE, "seed": 42,
        "python": platform.python_version(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "circuits": len(features), "runs": len(y),
        "timeouts": int(timeout.sum()), "validation_groups": len(set(groups)),
        "validation": "Fixed-recipe five-fold circuit-grouped experiment; full-data fit is not scored",
        "validation_seed_42": seed_results["42:bond_cost"]["metrics"],
        "validation_three_seed_summary": summary["bond_cost"],
        "validation_caveat": "Exploratory result; overall interval includes zero; long-success score declined.",
        "max_parse_seconds": max(r["featurize_seconds"] for r in timings),
        "mock_limits": {"gates": 20000, "source_chars": 8000000, "qubits": 512,
                        "expansion_visits": 100000, "simulation_seconds": 10,
                        "combined_soft_deadline_seconds": 14},
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (
            ROOT / "quantathon-harness/model.py", Path(__file__),
            ROOT / "quantathon-harness/train_improved.py", ROOT / "runtime-data.csv",
            EXPERIMENT / "features.json", EXPERIMENT / "summary.json",
            EXPERIMENT / "seed-results.json", EXPERIMENT / "provenance.json")}}
    artifact = {"schema_version": 3, "features": list(MOCK_FEATURE_NAMES),
                "pipeline": component, "log_bounds": [-9., max(math.log10(CAP_SECONDS), float(y.max()))],
                "training": metadata}

    # Check both schema compatibility and native/portable prediction fidelity
    # through the actual RuntimeModel API before publishing the candidate.
    previous_path = EXPERIMENT / "pre-integration"
    before_module = load_module("before_mock_model", previous_path / "model.py")
    before = before_module.RuntimeModel(previous_path)
    compatible = RuntimeModel(previous_path)
    legacy_delta = max(abs(before.predict(structural_cache[n]["features"], t) -
                           compatible.predict(structural_cache[n]["features"], t)) for n,t in keys)
    if legacy_delta != 0:
        raise RuntimeError("Predictions from the previous artifact changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mock-candidate-", dir=args.output.parent) as staging:
        path = Path(staging) / "runtime-model.json"
        path.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
        loaded = RuntimeModel(staging)
        portable, prediction_times = [], []
        for name, threshold in keys:
            start = time.perf_counter()
            portable.append(math.log10(loaded.predict(features[name], threshold)))
            prediction_times.append(time.perf_counter() - start)
        expected = np.clip(native, *artifact["log_bounds"])
        delta = float(np.max(np.abs(expected - portable)))
        if delta >= 1e-10 or max(prediction_times) >= 15:
            raise RuntimeError(f"Portable prediction check failed: difference={delta}")
        # Check arbitrary threshold evaluation and round-tripping the compact
        # featurizer result, beyond the three values in the training table.
        bell = loaded.featurize("OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];")
        roundtripped = json.loads(json.dumps(bell))
        for threshold in (1, 2, 32, 128, 1024):
            predicted = loaded.predict(bell, threshold)
            if not math.isfinite(predicted) or predicted <= 0 or predicted != loaded.predict(roundtripped, threshold):
                raise RuntimeError("Arbitrary threshold or feature JSON round-trip failed")
        for threshold in (0, -1, math.nan, math.inf):
            try:
                loaded.predict(bell, threshold)
            except ValueError:
                pass
            else:
                raise RuntimeError("Invalid threshold was accepted")
        try:
            loaded.predict(structural_cache[keys[0][0]]["features"], 64)
        except ValueError:
            pass
        else:
            raise RuntimeError("New artifact accepted missing mock features")
        metadata.update({"export_max_log10_difference": delta,
                         "max_predict_seconds": max(prediction_times)})
        path.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
        path.replace(args.output)

    with (args.report / "feature-timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
        writer.writeheader(); writer.writerows(timings)
    validation = {"artifact": str(args.output), "artifact_sha256": digest(args.output),
        "artifact_bytes": args.output.stat().st_size, "schema_version": 3, "features": len(MOCK_FEATURE_NAMES),
        "all_532_live_features_match_experiment": True, "feature_vectors_compared": len(features) * 3,
        "portable_predictions_compared": len(keys), "export_max_log10_difference": delta,
        "old_artifact_prediction_max_difference_seconds": legacy_delta,
        "max_featurize_seconds": metadata["max_parse_seconds"], "max_predict_seconds": max(prediction_times),
        "conservative_combined_seconds": metadata["max_parse_seconds"] + max(prediction_times),
        "feature_json_roundtrip": True, "arbitrary_thresholds": True, "invalid_thresholds_rejected": True,
        "missing_mock_features_rejected": True, "training_predictions_used_for_validation_score": False,
        "source_hashes": metadata["source_hashes"]}
    (args.report / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (args.report / "model-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
