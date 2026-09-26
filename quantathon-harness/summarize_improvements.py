"""Create a comparison figure and paired error tables from nested predictions."""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/model-improvement"


def load(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["filename"], int(float(row["threshold"]))


def main():
    comparison = json.loads((REPORT / "comparison.json").read_text())
    baseline = {key(r): float(r["pred_duration_s"]) for r in load(REPORT / "baseline-validation.csv")}
    improved = {key(r): float(r["pred_duration_s"]) for r in load(REPORT / "nested-validation.csv")}
    features = json.loads((REPORT / "baseline-features.json").read_text())
    rows, groups = [], {}
    for label in load(ROOT / "runtime-data.csv"):
        k = key(label)
        timeout = label["status"] == "timeout"
        target = 14400 if timeout else float(label["duration_s"])
        row = {"filename": k[0], "threshold": k[1], "status": label["status"], "target_s": target,
               "baseline_s": baseline[k], "improved_s": improved[k]}
        for name, prediction in (("baseline", baseline[k]), ("improved", improved[k])):
            row[name + "_factor"] = max(target / prediction, prediction / target)
            scored_prediction = min(prediction, 14400) if timeout else prediction
            row[name + "_score"] = max(0, 1 - abs(math.log10(scored_prediction / target)) / 2)
        row["score_change"] = row["improved_score"] - row["baseline_score"]
        signature = json.dumps(features[k[0]]["features"], sort_keys=True, separators=(",", ":"))
        group = hashlib.sha256(signature.encode()).hexdigest()
        aggregate = groups.setdefault(group, [0., 0])
        aggregate[0] += row["score_change"]
        aggregate[1] += 1
        rows.append(row)
    for suffix, sort_key in (("baseline-worst", "baseline_factor"), ("improved-worst", "improved_factor")):
        with (REPORT / f"paired-errors-{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r[sort_key], reverse=True))
    grouped = np.array(list(groups.values()))
    rng = np.random.default_rng(42)
    draws = rng.integers(len(grouped), size=(5000, len(grouped)))
    sampled = grouped[draws].sum(axis=1)
    interval = np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975])
    comparison["paired_group_bootstrap_score_gain_interval"] = interval.tolist()
    comparison["bootstrap_note"] = "Resamples feature groups of fixed paired predictions; does not retrain models or measure all training uncertainty."
    comparison["timeout_detection"] = {
        name: {"true_positives": sum(r["status"] == "timeout" and r[name + "_s"] >= 14399.999 for r in rows),
               "false_positives": sum(r["status"] != "timeout" and r[name + "_s"] >= 14399.999 for r in rows)}
        for name in ("baseline", "improved")}
    (REPORT / "comparison.json").write_text(json.dumps(comparison, indent=2))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=.76, bottom=.22, left=.07, right=.97, wspace=.25)
    fig.suptitle("Runtime model | baseline vs. nested model selection", x=.07, y=.965,
                 ha="left", fontsize=20, fontweight="bold")
    before, after = comparison["baseline"], comparison["nested_selected"]
    gain = (after["score"] - before["score"]) * 100
    fig.text(.07, .88, f"Challenge score: {before['score']:.2%} → {after['score']:.2%}   "
             f"({gain:+.2f} percentage points)     "
             f"R² seconds: {before['r2_seconds']:.3f} → {after['r2_seconds']:.3f}", fontsize=12)
    colors = ("#8596A7", "#008575")
    regimes = ["fast", "regular", "long_success", "timeout"]
    positions = np.arange(4)
    for i, (metrics, label) in enumerate(((before, "Baseline"), (after, "Improved pipeline"))):
        values = [100 * metrics["by_runtime"][k]["score"] for k in regimes]
        bars = axes[0].bar(positions + (i - .5) * .36, values, width=.36, label=label, color=colors[i])
        axes[0].bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=9, padding=3)
    axes[0].set_xticks(positions, ["<1 second", "1–1000 s", "≥1000 s\ncompleted", "Timeout"])
    axes[0].set(ylabel="Mean challenge score (%)", ylim=(0, 108))
    axes[0].set_title("Score by observed runtime", loc="left", fontweight="bold")
    axes[0].legend(loc="lower left")
    for name, label, color in zip(("baseline", "improved"), ("Baseline", "Improved pipeline"), colors):
        factors = np.sort([r[name + "_factor"] for r in rows])
        axes[1].step(factors, 100 * (1 - np.arange(1, len(factors) + 1) / len(factors)),
                     where="post", label=label, color=color, linewidth=2)
    factor_limit = max(2500, 1.25 * max(r[k] for r in rows for k in ("baseline_factor", "improved_factor")))
    axes[1].set(xscale="log", xlim=(1, factor_limit), ylim=(0, 50),
                xlabel="Multiplicative error", ylabel="Runs with larger error (%)")
    axes[1].set_title("Frequency of large errors", loc="left", fontweight="bold")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.07, .10, "Same 5 outer circuit-grouped folds. Improved model/routing selection uses 3 inner folds; the final artifact is fitted on all labels.", fontsize=9)
    fig.text(.07, .065, "Timeout reference = 14,400 s. The library was previously inspected for feature design; this is not an untouched external benchmark.", fontsize=9)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(REPORT / f"comparison.{extension}", dpi=170, facecolor="white")
    plt.close(fig)
    print(json.dumps({"gain_percentage_points": gain, "gain_interval": (interval * 100).tolist(),
                      "timeout_detection": comparison["timeout_detection"]}, indent=2))


if __name__ == "__main__":
    main()
