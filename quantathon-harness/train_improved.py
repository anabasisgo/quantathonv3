"""Compare runtime models with nested circuit-grouped validation.

Training dependencies: scikit-learn==1.9.1 and xgboost-cpu==3.4.1.
Ask for installation approval before using uv to install missing packages.
Inference remains standard-library-only and stores all fitted parameters in JSON.
"""

import argparse
import csv
import hashlib
import inspect
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import model as runtime_module

from model import (BASE_FEATURE_NAMES, FEATURE_NAMES, RuntimeModel, CAP_SECONDS,
                   _feature_vector, _predict_component, _calibration_features)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/model-improvement"
BASE_INDEXES = [FEATURE_NAMES.index(name) for name in BASE_FEATURE_NAMES]
LOG_CAP = math.log10(CAP_SECONDS)
REGRESSORS = ["baseline", "enhanced", "fine", "absolute", "squared", "quantile", "survival"]
TRAINING_FAMILIES = REGRESSORS + ["routed_0.5", "routed_1", "routed_2", "calibrated"]


def feature_fingerprint():
    functions = [runtime_module._sample_program, runtime_module._gate_size,
                 runtime_module._numeric_angle, runtime_module._gate_profile,
                 RuntimeModel.featurize]
    text = "\n".join(inspect.getsource(f) for f in functions)
    text += repr(FEATURE_NAMES)
    text += repr((runtime_module._SAMPLE_CHARS, runtime_module._SAMPLE_WINDOWS,
                  runtime_module._DEFINITION_LIMIT))
    for name in ("_COMMENTS", "_DEFINITION", "_DECLARATION", "_STATEMENT", "_OPERATION", "_OPERAND", "_CONDITION"):
        text += getattr(runtime_module, name).pattern
    for name in ("_IGNORED", "_GATES", "_KNOWN", "_DIAGONAL", "_CLIFFORD", "_ROTATIONS"):
        text += repr(sorted(getattr(runtime_module, name)))
    return hashlib.sha256(text.encode()).hexdigest()


def experiment_fingerprint(x, y, groups, timeout):
    digest = hashlib.sha256()
    for values in (x, y, groups, timeout):
        digest.update(values.tobytes())
    for function in (make_extra, fit_family, predict_family, apply_recipe, recipe_options, score, select_recipe):
        digest.update(inspect.getsource(function).encode())
    digest.update(repr(TRAINING_FAMILIES).encode())
    digest.update(f"{sklearn.__version__}:{xgb.__version__}".encode())
    return digest.hexdigest()


def score(y, pred, timeout):
    scored = np.where(timeout, np.minimum(pred, LOG_CAP), pred)
    return np.maximum(0, 1 - np.abs(scored - y) / 2)


def evaluate(y, pred, timeout):
    actual, predicted = 10**y, 10**pred
    factor = 10 ** np.abs(pred - y)
    groups = {"fast": (~timeout) & (actual < 1),
              "regular": (~timeout) & (actual >= 1) & (actual < 1000),
              "long_success": (~timeout) & (actual >= 1000), "timeout": timeout}
    result = {"score": float(score(y, pred, timeout).mean()),
              "r2_seconds": float(1 - np.sum((actual - predicted)**2) / np.sum((actual - actual.mean())**2)),
              "r2_log10_seconds": float(1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)),
              "log10_mae": float(np.abs(y - pred).mean()),
              "median_error_factor": float(np.median(factor)),
              "errors_over_10x": int(np.sum(factor > 10)),
              "timeouts_predicted_at_cap": int(np.sum(timeout & (pred >= LOG_CAP - 1e-8)))}
    result["by_runtime"] = {name: {"runs": int(mask.sum()),
                                   "score": float(score(y[mask], pred[mask], timeout[mask]).mean()),
                                   "median_error_factor": float(np.median(factor[mask]))}
                            for name, mask in groups.items() if mask.any()}
    return result


def recipe_options():
    options = [{"family": name} for name in REGRESSORS + ["calibrated"]]
    for boundary in (0.5, 1, 2):
        for confidence in (.65, .85, .95):
            for timeout_confidence in (.35, .55, .75, .95):
                options.append({"family": f"routed_{boundary}", "fast_confidence": confidence,
                                "regular_confidence": confidence,
                                "timeout_confidence": timeout_confidence})
    return options


def make_extra(fine=False):
    return ExtraTreesRegressor(n_estimators=200, max_depth=None if fine else 14,
                               min_samples_leaf=1 if fine else 2,
                               max_features=1.0 if fine else .85, random_state=42, n_jobs=2)


def fit_family(family, x, y, groups, timeout):
    if family.startswith("routed_"):
        boundary = float(family.split("_")[1])
        classes = np.where(timeout, 2, np.where(y < math.log10(boundary), 0, 1))
        classifier = ExtraTreesClassifier(n_estimators=160, max_depth=16,
                                          min_samples_leaf=2, max_features=.9,
                                          class_weight="balanced", random_state=42, n_jobs=2)
        classifier.fit(x, classes)
        fitted = {"family": family, "classifier": classifier, "base": make_extra(True).fit(x, y)}
        for label, name in ((0, "fast"), (1, "regular")):
            mask = classes == label
            fitted[name] = make_extra(True).fit(x[mask], y[mask]) if mask.sum() >= 4 else fitted["base"]
        return fitted
    if family == "calibrated":
        crossfit = np.zeros(len(y))
        for train, valid in GroupKFold(3, shuffle=True, random_state=123).split(x, y, groups):
            crossfit[valid] = make_extra(True).fit(x[train], y[train]).predict(x[valid])
        design = np.array([_calibration_features(p, t) for p, t in zip(crossfit, x[:, -1])])
        penalty = np.eye(design.shape[1]) * 10
        penalty[0, 0] = .1
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ (y - crossfit))
        return {"family": family, "base": make_extra(True).fit(x, y), "coefficients": coefficients}
    indexes = BASE_INDEXES if family == "baseline" else list(range(x.shape[1]))
    xx = x[:, indexes]
    if family in ("baseline", "enhanced", "fine"):
        estimator = make_extra(family == "fine").fit(xx, y)
    elif family in ("absolute", "squared", "quantile"):
        loss = {"absolute": "absolute_error", "squared": "squared_error", "quantile": "quantile"}[family]
        estimator = HistGradientBoostingRegressor(
            loss=loss, quantile=.65 if family == "quantile" else None,
            learning_rate=.07, max_iter=250, max_leaf_nodes=15,
            min_samples_leaf=10, l2_regularization=.5, early_stopping=False, random_state=42).fit(xx, y)
    elif family == "survival":
        data = xgb.DMatrix(xx)
        data.set_float_info("label_lower_bound", 10**y)
        data.set_float_info("label_upper_bound", np.where(timeout, np.inf, 10**y))
        estimator = xgb.train({"objective": "survival:aft", "eval_metric": "aft-nloglik",
                               "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
                               "tree_method": "hist", "max_depth": 4, "eta": .05,
                               "min_child_weight": 5, "lambda": 2, "seed": 42, "nthread": 2},
                              data, num_boost_round=300)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return {"family": family, "estimator": estimator, "indexes": indexes}


def predict_family(fitted, x):
    family = fitted["family"]
    if family.startswith("routed_"):
        probabilities = np.zeros((len(x), 3))
        probabilities[:, fitted["classifier"].classes_] = fitted["classifier"].predict_proba(x)
        return {**{name: fitted[name].predict(x) for name in ("base", "fast", "regular")},
                "probabilities": probabilities}
    if family == "calibrated":
        base = fitted["base"].predict(x)
        design = np.array([_calibration_features(p, t) for p, t in zip(base, x[:, -1])])
        return base + np.clip(design @ fitted["coefficients"], -.5, .5)
    xx = x[:, fitted["indexes"]]
    if family == "survival":
        return fitted["estimator"].predict(xgb.DMatrix(xx), output_margin=True).astype(float) / math.log(10)
    return fitted["estimator"].predict(xx)


def apply_recipe(raw, recipe):
    if not recipe["family"].startswith("routed_"):
        return raw
    prediction = raw["base"].copy()
    p = raw["probabilities"]
    regular = p[:, 1] >= recipe["regular_confidence"]
    fast = p[:, 0] >= recipe["fast_confidence"]
    timeout = p[:, 2] >= recipe["timeout_confidence"]
    prediction[regular] = raw["regular"][regular]
    prediction[fast] = raw["fast"][fast]
    prediction[timeout] = LOG_CAP
    return prediction


def select_recipe(x, y, groups, timeout, n_splits, label):
    options = recipe_options()
    splits = list(GroupKFold(n_splits, shuffle=True, random_state=2026).split(x, y, groups))
    scored = []
    for family in TRAINING_FAMILIES:
        raw = None
        for train, valid in splits:
            fitted = fit_family(family, x[train], y[train], groups[train], timeout[train])
            current = predict_family(fitted, x[valid])
            if isinstance(current, dict):
                if raw is None:
                    raw = {k: np.zeros((len(y),) + v.shape[1:]) for k, v in current.items()}
                for k in raw:
                    raw[k][valid] = current[k]
            else:
                if raw is None: raw = np.zeros(len(y))
                raw[valid] = current
        family_scores = []
        for recipe in options:
            if recipe["family"] == family:
                prediction = apply_recipe(raw, recipe)
                entry = {"recipe": recipe, "score": float(score(y, prediction, timeout).mean())}
                scored.append(entry)
                family_scores.append(entry["score"])
        print(f"{label}: {family} inner score {max(family_scores):.4%}", flush=True)
    return max(scored, key=lambda entry: entry["score"])["recipe"], scored


def export_estimator(estimator, indexes=None, x_reference=None):
    result = {"kind": "forest", "bias": 0., "trees": []}
    if indexes is not None:
        result["feature_indices"] = list(indexes)
    if isinstance(estimator, xgb.Booster):
        result["comparison"] = "lt"
        for dumped in estimator.get_dump(dump_format="json"):
            root, nodes = json.loads(dumped), []

            def walk(node):
                index = len(nodes)
                nodes.append(None)
                if "leaf" in node:
                    nodes[index] = [-1, node["leaf"] / math.log(10)]
                else:
                    children = {child["nodeid"]: child for child in node["children"]}
                    left, right = walk(children[node["yes"]]), walk(children[node["no"]])
                    nodes[index] = [int(node["split"].removeprefix("f")), node["split_condition"], left, right]
                return index

            walk(root)
            result["trees"].append(nodes)
        reference = x_reference[0].tolist()
        native = float(estimator.predict(xgb.DMatrix(x_reference[:1]), output_margin=True)[0]) / math.log(10)
        # Recover the fitted intercept, including XGBoost's automatic base score.
        local = dict(result)
        local.pop("feature_indices", None)
        result["bias"] = native - _predict_component(local, reference)
    elif isinstance(estimator, HistGradientBoostingRegressor):
        result["bias"] = float(estimator._baseline_prediction.ravel()[0])
        for iteration in estimator._predictors:
            nodes = []
            for node in iteration[0].nodes:
                if node["is_leaf"]:
                    nodes.append([-1, float(node["value"])])
                else:
                    nodes.append([int(node["feature_idx"]), float(node["num_threshold"]),
                                  int(node["left"]), int(node["right"])])
            result["trees"].append(nodes)
    else:
        classification = isinstance(estimator, ExtraTreesClassifier)
        weight = 1 / len(estimator.estimators_)
        if classification:
            result["bias"] = [0., 0., 0.]
        for tree_estimator in estimator.estimators_:
            tree, nodes = tree_estimator.tree_, []
            for i in range(tree.node_count):
                if tree.children_left[i] < 0:
                    if classification:
                        value = [0., 0., 0.]
                        probabilities = tree.value[i, 0] / tree.value[i, 0].sum()
                        for label, probability in zip(estimator.classes_, probabilities):
                            value[int(label)] = float(probability) * weight
                    else:
                        value = float(tree.value[i, 0, 0]) * weight
                    nodes.append([-1, value])
                else:
                    nodes.append([int(tree.feature[i]), float(tree.threshold[i]),
                                  int(tree.children_left[i]), int(tree.children_right[i])])
            result["trees"].append(nodes)
    return result


def export_fitted(fitted, recipe, x):
    family = fitted["family"]
    if family.startswith("routed_"):
        return {"kind": "routed", **{k: v for k, v in recipe.items() if k != "family"},
                "classifier": export_estimator(fitted["classifier"]),
                **{name: export_estimator(fitted[name]) for name in ("base", "fast", "regular")}}
    if family == "calibrated":
        return {"kind": "calibrated", "base": export_estimator(fitted["base"]),
                "coefficients": fitted["coefficients"].tolist(), "clip": .5,
                "threshold_index": FEATURE_NAMES.index("log2_threshold")}
    return export_estimator(fitted["estimator"], fitted["indexes"], x[:, fitted["indexes"]])


def write_predictions(path, keys, prediction):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team", "filename", "threshold", "pred_duration_s"])
        writer.writerows(("cross-validation", name, threshold, 10**float(p))
                         for (name, threshold), p in zip(keys, prediction))


def load_data(labels, report):
    from run import read_qasm
    baseline = json.loads((report / "baseline-features.json").read_text())
    cache_path = report / "enhanced-features.json"
    if not cache_path.exists():
        parser, cached = RuntimeModel(load_model=False), {}
        for name in sorted(baseline):
            source_path = ROOT / "training_circuits" / (name + ".zst")
            source = read_qasm(source_path)
            start = time.perf_counter()
            features = parser.featurize(source)
            elapsed = time.perf_counter() - start
            if elapsed > 15:
                raise RuntimeError(f"Parser exceeded limit: {name}")
            cached[name] = {"features": features, "parse_seconds": elapsed,
                            "compressed_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
        cache_path.write_text(json.dumps({"feature_fingerprint": feature_fingerprint(), "circuits": cached}))
    cache = json.loads(cache_path.read_text())
    if cache.get("feature_fingerprint") != feature_fingerprint():
        raise ValueError("Feature cache does not match the current parser; regenerate enhanced-features.json")
    enhanced = cache["circuits"]
    for name, item in enhanced.items():
        source = ROOT / "training_circuits" / (name + ".zst")
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["compressed_sha256"]:
            raise ValueError(f"Circuit changed since feature extraction: {name}")
    x, y, groups, timeout, keys = [], [], [], [], []
    with labels.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        name = row["filename"].strip()
        threshold = int(float(row["threshold"]))
        is_timeout = row["status"].strip().lower() == "timeout"
        duration = CAP_SECONDS if is_timeout else float(row["duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid target for {name}")
        x.append(_feature_vector(enhanced[name]["features"], threshold))
        signature = json.dumps(baseline[name]["features"], sort_keys=True, separators=(",", ":"))
        groups.append(hashlib.sha256(signature.encode()).hexdigest())
        y.append(math.log10(duration))
        timeout.append(is_timeout)
        keys.append((name, threshold))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate label keys")
    return (np.asarray(x, dtype=np.float32), np.asarray(y), np.asarray(groups),
            np.asarray(timeout), keys, enhanced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "runtime-data.csv")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=ROOT / "quantathon-harness/artifacts/runtime-model.json")
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    x, y, groups, timeout, keys, features = load_data(args.labels, args.report)
    fingerprint = experiment_fingerprint(x, y, groups, timeout)
    selected_oof, baseline_oof = np.zeros(len(y)), np.zeros(len(y))
    diagnostics = {name: np.zeros(len(y)) for name in TRAINING_FAMILIES}
    fold_results = []
    splits = list(GroupKFold(5, shuffle=True, random_state=42).split(x, y, groups))
    for fold, (train, valid) in enumerate(splits):
        fold_number = fold + 1
        checkpoint = args.report / f"fold-{fold_number}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("experiment_fingerprint") != fingerprint or saved["valid_indexes"] != valid.tolist():
                raise ValueError("Checkpoint does not match the current data or experiment configuration")
            selected_oof[valid] = saved["selected_predictions"]
            baseline_oof[valid] = saved["baseline_predictions"]
            for family in diagnostics:
                diagnostics[family][valid] = saved["diagnostic_predictions"][family]
            fold_results.append(saved)
            print(f"Loaded completed fold {fold_number}", flush=True)
            continue
        recipe, inner = select_recipe(x[train], y[train], groups[train], timeout[train], 3, f"Outer {fold_number}")
        for family in TRAINING_FAMILIES:
            family_recipe = max((r for r in inner if r["recipe"]["family"] == family), key=lambda r: r["score"])["recipe"]
            fitted = fit_family(family, x[train], y[train], groups[train], timeout[train])
            predicted = apply_recipe(predict_family(fitted, x[valid]), family_recipe)
            diagnostics[family][valid] = predicted
            if family == recipe["family"]:
                selected_oof[valid] = apply_recipe(predict_family(fitted, x[valid]), recipe)
            if family == "baseline":
                baseline_oof[valid] = predicted
        saved = {"fold": fold_number, "experiment_fingerprint": fingerprint, "recipe": recipe, "inner_results": inner,
                 "valid_indexes": valid.tolist(), "selected_predictions": selected_oof[valid].tolist(),
                 "baseline_predictions": baseline_oof[valid].tolist(),
                 "diagnostic_predictions": {k: v[valid].tolist() for k, v in diagnostics.items()},
                 "selected_metrics": evaluate(y[valid], selected_oof[valid], timeout[valid]),
                 "baseline_metrics": evaluate(y[valid], baseline_oof[valid], timeout[valid])}
        checkpoint.write_text(json.dumps(saved, indent=2))
        fold_results.append(saved)
        print(f"OUTER {fold_number}: selected {recipe}, score={saved['selected_metrics']['score']:.4%}, "
              f"baseline={saved['baseline_metrics']['score']:.4%}", flush=True)
    summary = {"baseline": evaluate(y, baseline_oof, timeout), "nested_selected": evaluate(y, selected_oof, timeout),
               "diagnostics": {k: evaluate(y, p, timeout) for k, p in diagnostics.items()},
               "outer_fold_recipes": [r["recipe"] for r in fold_results],
               "evaluation": "5 outer circuit-grouped folds; 3 inner grouped folds for model/routing selection",
               "caveat": "This library was previously inspected for feature design; outer folds isolate fitting and tuning, not prior human analysis."}
    (args.report / "comparison.json").write_text(json.dumps(summary, indent=2))
    write_predictions(args.report / "nested-validation.csv", keys, selected_oof)
    write_predictions(args.report / "baseline-validation.csv", keys, baseline_oof)
    for family, prediction in diagnostics.items():
        write_predictions(args.report / f"validation-{family}.csv", keys, prediction)
    print("NESTED COMPARISON", json.dumps(summary, indent=2), flush=True)
    recipe, final_selection = select_recipe(x, y, groups, timeout, 5, "Final selection")
    fitted = fit_family(recipe["family"], x, y, groups, timeout)
    component = export_fitted(fitted, recipe, x)
    metadata = {"model": "nested_selected", "recipe": recipe, "seed": 42,
                "sklearn_version": sklearn.__version__, "xgboost_version": xgb.__version__,
                "circuits": len(features), "runs": len(y), "timeouts": int(timeout.sum()),
                "validation_groups": len(set(groups)), "validation": summary["evaluation"],
                "candidates": {"nested_selected": summary["nested_selected"]},
                "baseline_metrics": summary["baseline"], "final_selection": final_selection,
                "max_parse_seconds": max(v["parse_seconds"] for v in features.values()),
                "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                "validation_caveat": summary["caveat"]}
    metadata["experiment_fingerprint"] = fingerprint
    artifact = {"schema_version": 2, "features": list(FEATURE_NAMES), "pipeline": component,
                "log_bounds": [-9., max(LOG_CAP, float(y.max()))], "training": metadata}
    # Export fidelity is part of training: prevent deploying a different predictor.
    native = apply_recipe(predict_family(fitted, x), recipe)
    portable = np.array([_predict_component(component, row.tolist()) for row in x])
    delta = float(np.max(np.abs(native - portable)))
    if not np.allclose(native, portable, rtol=0, atol=2e-5):
        raise RuntimeError(f"Portable export differs from native model by {delta}")
    metadata["export_max_log10_difference"] = delta
    artifact_path = args.report / "selected-artifact.json"
    artifact_path.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
    summary["final_recipe"] = recipe
    summary["export_max_log10_difference"] = delta
    summary["promoted"] = summary["nested_selected"]["score"] > summary["baseline"]["score"]
    if summary["promoted"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact_path, args.output)
    (args.report / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(f"Final recipe: {recipe}; promoted={summary['promoted']}; export difference={delta}", flush=True)


if __name__ == "__main__":
    main()
