"""Fit the bounded-probe ExtraTrees model from committed training data."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from probe_model import FEATURE_NAMES, MAX_PROBE_OPS, vector

CAP_SECONDS = 14400.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fit(features_path, labels_path, output_path):
    features = {r["filename"]: r for r in read_csv(features_path)}
    labels = read_csv(labels_path)
    if len(features) != 532 or len(labels) != 1497:
        raise ValueError("Training table counts changed; inspect before retraining")
    x, y = [], []
    for row in labels:
        feat = features[row["filename"]]
        if not int(feat["probe_usable"]) or float(feat["ops"]) >= MAX_PROBE_OPS:
            continue
        threshold = int(row["threshold"])
        x.append(vector(feat, feat, threshold))
        actual = CAP_SECONDS if row["status"] == "timeout" else float(row["duration_s"])
        y.append(math.log10(actual))
    x, y = np.asarray(x), np.asarray(y)
    if x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid training feature matrix")
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                max_features=0.9, random_state=20260926, n_jobs=-1)
    model.fit(x, y)
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append({
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "left": tree.children_left.tolist(),
            "right": tree.children_right.tolist(),
            "value": tree.value[:, 0, 0].tolist(),
        })
    # Check the portable traversal before writing the deployment artifact.
    for row, expected in zip(x[:20], model.predict(x[:20])):
        values = []
        for tree in trees:
            node = 0
            while tree["feature"][node] >= 0:
                feature = tree["feature"][node]
                node = tree["left"][node] if row[feature] <= tree["threshold"][node] else tree["right"][node]
            values.append(tree["value"][node])
        if abs(sum(values) / len(values) - expected) > 1e-9:
            raise AssertionError("Portable tree inference differs from scikit-learn")
    artifact = {
        "kind": "probe-extra-trees-v1",
        "features": list(FEATURE_NAMES),
        "training_rows": len(x),
        "training_circuits": len({r["filename"] for r in labels if int(features[r["filename"]]["probe_usable"]) and float(features[r["filename"]]["ops"]) < MAX_PROBE_OPS}),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "trees": trees,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(artifact, stream, separators=(",", ":"))
    print(f"trained {len(x)} rows, {len(trees)} trees; wrote {output_path}")


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path(__file__).resolve().parent / "training_features.csv")
    parser.add_argument("--labels", type=Path, default=repo / "runtime-data.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts/probe-extra-trees.json.gz")
    args = parser.parse_args()
    fit(args.features, args.labels, args.out)


if __name__ == "__main__":
    main()
