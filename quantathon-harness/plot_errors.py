"""Plot saved out-of-fold runtime predictions without retraining the model.

Run from the repository root after approving installation of matplotlib:
    uv run --with matplotlib python quantathon-harness/plot_errors.py \
        --predictions /tmp/quantathon-validation.csv

All plotted residuals use unmodified predictions. Timeout targets are 14,400
seconds, a lower bound rather than an observed completion time. Only challenge
scores cap predictions above 14,400 seconds on timeout rows, as score.py does.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAP = 14400.0
COLORS = {16: "#0072B2", 64: "#D89000", 512: "#009E73"}
ACTUAL_COLOR = "#244A73"
PREDICTED_COLOR = "#C35132"
TIMEOUT_NOTE = (
    "Triangles / * = timeout: plotted at the 14,400 s lower bound. "
    "Predictions are unclipped; timeout residuals are relative to that bound."
)


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return row["filename"].strip(), int(float(row["threshold"]))


def join_predictions(labels, predictions):
    by_key = {}
    for row in predictions:
        key = row_key(row)
        if key in by_key:
            raise ValueError(f"Duplicate prediction: {key}")
        if row.get("team") != "cross-validation":
            raise ValueError("Expected held-out predictions with team=cross-validation")
        by_key[key] = float(row["pred_duration_s"])
    result, seen = [], set()
    for row in labels:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate label: {key}")
        seen.add(key)
        timed_out = row["status"].strip().lower() == "timeout"
        target = CAP if timed_out else float(row["duration_s"])
        predicted = by_key[key]
        if not all(math.isfinite(v) and v > 0 for v in (target, predicted)):
            raise ValueError(f"Nonpositive or nonfinite duration for {key}")
        ratio = predicted / target
        scored_prediction = min(predicted, CAP) if timed_out else predicted
        result.append({
            "filename": key[0], "threshold": key[1], "status": row["status"],
            "target_duration_s": target, "pred_duration_s": predicted,
            "error_s": predicted - target,
            "absolute_error_s": abs(predicted - target),
            "prediction_ratio": ratio, "error_factor": max(ratio, 1 / ratio),
            "log10_ratio": math.log10(ratio),
            "challenge_score": max(0, 1 - abs(math.log10(scored_prediction / target)) / 2),
        })
    if set(by_key) != seen:
        raise ValueError("Prediction and label keys must match exactly")
    return result


def metrics(rows):
    actual = np.array([r["target_duration_s"] for r in rows])
    predicted = np.array([r["pred_duration_s"] for r in rows])
    factor = np.array([r["error_factor"] for r in rows])

    def r_squared(y, p):
        denominator = np.sum((y - np.mean(y)) ** 2)
        return float(1 - np.sum((y - p) ** 2) / denominator) if denominator else None

    return {
        "runs": len(rows), "circuits": len({r["filename"] for r in rows}),
        "timeouts": sum(r["status"] == "timeout" for r in rows),
        "r2_seconds": r_squared(actual, predicted),
        "r2_log10_seconds": r_squared(np.log10(actual), np.log10(predicted)),
        "challenge_score": float(np.mean([r["challenge_score"] for r in rows])),
        "log10_mae": float(np.mean(np.abs(np.log10(predicted / actual)))),
        "mae_seconds": float(np.mean(np.abs(predicted - actual))),
        "rmse_seconds": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "median_error_factor": float(np.median(factor)),
        "p90_error_factor": float(np.quantile(factor, .90)),
        "p95_error_factor": float(np.quantile(factor, .95)),
        "errors_greater_than_2x": int(np.sum(factor > 2)),
        "errors_greater_than_10x": int(np.sum(factor > 10)),
        "errors_greater_than_100x": int(np.sum(factor > 100)),
    }


def save_figure(fig, output, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=170, facecolor="white")
    plt.close(fig)


def scatter_by_threshold(ax, rows, x_key, y_key):
    for threshold in sorted({r["threshold"] for r in rows}):
        for timed_out, marker in ((False, "o"), (True, "^")):
            selected = [r for r in rows if r["threshold"] == threshold
                        and (r["status"] == "timeout") == timed_out]
            ax.scatter([r[x_key] for r in selected], [r[y_key] for r in selected],
                       color=COLORS.get(threshold, "#777777"), marker=marker,
                       s=42 if timed_out else 17, alpha=.9 if timed_out else .45,
                       linewidths=.4 if timed_out else 0,
                       edgecolors="white" if timed_out else "none",
                       label=f"Threshold {threshold}" if not timed_out else None)


def plot_overview(rows, summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.6))
    fig.subplots_adjust(top=.83, bottom=.10, left=.08, right=.97, hspace=.40, wspace=.28)
    fig.suptitle("Runtime model | held-out prediction errors", x=.08, y=.975,
                 ha="left", fontsize=22, fontweight="bold")
    fig.text(.08, .932, "5-fold validation grouped by circuit features · "
             f"{summary['runs']:,} runs · {summary['circuits']} circuits · "
             + summary.get("model_description", "200 Extra Trees"),
             fontsize=11, color="#536171")
    fig.text(.08, .89, f"R² seconds  {summary['r2_seconds']:.3f}     "
             f"R² log-seconds  {summary['r2_log10_seconds']:.3f}     "
             f"Challenge score  {summary['challenge_score']:.2%}     "
             f"Errors >10×  {summary['errors_greater_than_10x']}", fontsize=12)

    ax = axes[0, 0]
    scatter_by_threshold(ax, rows, "target_duration_s", "pred_duration_s")
    values = [r[key] for r in rows for key in ("target_duration_s", "pred_duration_s")]
    limits = (min(values) / 1.8, max(values) * 1.8)
    line = np.geomspace(*limits, 200)
    ax.fill_between(line, line / 2, line * 2, color="#BAC5D1", alpha=.18)
    ax.plot(line, line, color="#657285", linewidth=1, linestyle="--")
    ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
           xlabel="Actual runtime / timeout reference (seconds)", ylabel="Predicted runtime (seconds)")
    ax.set_title("A  Actual vs. predicted", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.text(.97, .04, "Dashed: exact · shaded: within 2×", transform=ax.transAxes,
            ha="right", fontsize=9, color="#536171")

    ax = axes[0, 1]
    scatter_by_threshold(ax, rows, "target_duration_s", "log10_ratio")
    ax.axhspan(-math.log10(2), math.log10(2), color="#BAC5D1", alpha=.18)
    for y in (-1, 0, 1):
        ax.axhline(y, color="#657285", linewidth=.8, linestyle="-" if y == 0 else "--")
    extent = max(3.6, max(abs(r["log10_ratio"]) for r in rows) + .25)
    ticks = list(range(-int(extent), int(extent) + 1))
    ax.set(xscale="log", xlabel="Actual runtime / timeout reference (seconds)",
           ylabel="Prediction relative to actual", ylim=(-extent, extent))
    ax.set_yticks(ticks, ["Exact" if t == 0 else f"{10**abs(t):g}× {'low' if t < 0 else 'high'}" for t in ticks])
    ax.set_title("B  Direction and size of the errors", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for threshold in sorted({r["threshold"] for r in rows}):
        factors = np.sort([r["error_factor"] for r in rows if r["threshold"] == threshold])
        survival = 100 * (1 - np.arange(1, len(factors) + 1) / len(factors))
        ax.step(np.r_[1, factors], np.r_[100, survival], where="post",
                color=COLORS.get(threshold, "#777777"), linewidth=2, label=f"Threshold {threshold}")
    for factor in (2, 10):
        ax.axvline(factor, color="#8995A3", linestyle="--", linewidth=.8)
    ax.set(xscale="log", xlabel="Error factor (larger = worse)",
           ylabel="Runs with a larger error", ylim=(0, 102),
           xlim=(1, max(2500, 1.25 * max(r["error_factor"] for r in rows))))
    ax.set_xticks([1, 2, 5, 10, 100, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("C  How common are large errors?", loc="left", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    actual = np.array([r["target_duration_s"] for r in rows])
    errors = np.array([r["error_s"] for r in rows])
    bins = np.digitize(actual, [1, 10, 100, 1000])
    squared = errors ** 2
    total = squared.sum()
    low = [100 * squared[(bins == i) & (errors < 0)].sum() / total for i in range(5)]
    high = [100 * squared[(bins == i) & (errors >= 0)].sum() / total for i in range(5)]
    ax.bar(range(5), low, color=PREDICTED_COLOR, label="Underestimates")
    ax.bar(range(5), high, bottom=low, color=ACTUAL_COLOR, label="Overestimates")
    ax.set_xticks(range(5), ["<1 s", "1–10 s", "10–100 s", "100–1000 s", "≥1000 s"])
    for i in range(5):
        ax.text(i, low[i] + high[i] + 2, f"n={np.sum(bins == i)}", ha="center", fontsize=9)
    ax.set(xlabel="Actual runtime / timeout reference", ylabel="Share of total squared error", ylim=(0, 108))
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("D  Which runs drive the error in seconds?", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    fig.text(.08, .044, TIMEOUT_NOTE, fontsize=9, color="#536171")
    fig.text(.08, .023, summary.get("validation_note", "These folds were also used to select the model. Results are validation estimates, not an independent final test."),
             fontsize=9, color="#536171")
    save_figure(fig, output, "error-overview")


def plot_worst_cases(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.8))
    fig.subplots_adjust(top=.79, bottom=.13, left=.12, right=.93, wspace=.73)
    fig.suptitle("Runtime model | the largest held-out misses", x=.04, y=.965,
                 ha="left", fontsize=22, fontweight="bold")
    handles = [Line2D([], [], marker="o", linestyle="none", color=ACTUAL_COLOR, label="Actual / timeout reference"),
               Line2D([], [], marker="o", linestyle="none", color=PREDICTED_COLOR, label="Predicted")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.035, .92), ncol=2, frameon=False)
    for ax, key, title in zip(axes, ("error_factor", "absolute_error_s"),
                              ("A  Worst proportional errors", "B  Worst errors in seconds")):
        selected = sorted(rows, key=lambda r: r[key], reverse=True)[:12]
        for y, row in enumerate(selected):
            target, pred = row["target_duration_s"], row["pred_duration_s"]
            ax.plot([target, pred], [y, y], color="#BCC5CF", linewidth=2, zorder=1)
            ax.scatter(target, y, color=ACTUAL_COLOR, s=48,
                       marker="^" if row["status"] == "timeout" else "o", zorder=3)
            ax.scatter(pred, y, color=PREDICTED_COLOR, s=40, zorder=3)
            value = f"{row[key]:,.0f}×" if key == "error_factor" else f"{row[key]:,.0f} s"
            ax.text(1.025, y, value, transform=ax.get_yaxis_transform(), va="center", fontsize=10)
        ax.set_yticks(range(len(selected)),
                      [f"{r['filename'].removesuffix('.qasm')} · t={r['threshold']}"
                       + (" *" if r["status"] == "timeout" else "") for r in selected])
        minimum = min(min(r["target_duration_s"], r["pred_duration_s"]) for r in selected)
        maximum = max(max(r["target_duration_s"], r["pred_duration_s"]) for r in selected)
        ax.set(xscale="log", xlim=(min(.2, minimum / 2), max(100000, maximum * 2)), ylim=(len(selected) - .4, -.6),
               xlabel="Runtime (seconds, logarithmic scale)")
        ax.set_title(title, loc="left", fontweight="bold", pad=15)
        ax.text(1.025, 1.025, "Error", transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.grid(axis="y", visible=False)
    fig.text(.04, .055, "Each row is one circuit and threshold; the same circuit can fail at several thresholds.",
             fontsize=10, color="#536171")
    fig.text(.04, .028, TIMEOUT_NOTE, fontsize=9, color="#536171")
    save_figure(fig, output, "worst-cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=ROOT / "runtime-data.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "quantathon-harness/artifacts/runtime-model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/runtime-errors")
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    rows = join_predictions(load_csv(args.labels), load_csv(args.predictions))
    summary = metrics(rows)
    if model["schema_version"] == 2:
        summary["model_description"] = "Nested selection of regression / routing models"
        summary["validation_note"] = "Model and routing choices used inner grouped folds. This circuit library was previously inspected during feature design."
    expected = model["training"]["candidates"][model["training"]["model"]]
    canonical_labels = args.labels.read_bytes().replace(b"\r\n", b"\n")
    hashes = {hashlib.sha256(canonical_labels).hexdigest(),
              hashlib.sha256(canonical_labels.replace(b"\n", b"\r\n")).hexdigest()}
    if model["training"]["labels_sha256"] not in hashes:
        raise ValueError("Labels do not match the artifact's training-label hash")
    for key in ("challenge_score", "log10_mae"):
        if not math.isclose(summary[key], expected["score" if key == "challenge_score" else key], abs_tol=1e-10):
            raise ValueError(f"Saved predictions do not match the recorded validation {key}")
    summary["by_threshold"] = {str(t): metrics([r for r in rows if r["threshold"] == t])
                               for t in sorted({r["threshold"] for r in rows})}
    summary["long_runs_share_of_squared_error"] = (
        sum(r["error_s"] ** 2 for r in rows if r["target_duration_s"] >= 1000)
        / sum(r["error_s"] ** 2 for r in rows))
    summary["provenance"] = {
        "evaluation": model["training"]["validation"],
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "timeout_handling": TIMEOUT_NOTE,
        "model_selection_caveat": model["training"].get("validation_caveat", "Validation folds were also used to select among three candidate models."),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, key in (("errors-by-factor.csv", "error_factor"), ("errors-by-seconds.csv", "absolute_error_s")):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r[key], reverse=True))
    prediction_copy = args.output / "validation-predictions.csv"
    if args.predictions.resolve() != prediction_copy.resolve():
        shutil.copyfile(args.predictions, prediction_copy)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    plot_overview(rows, summary, args.output)
    plot_worst_cases(rows, args.output)
    print(json.dumps(summary, indent=2))
    print(f"Plots and error tables written to {args.output}")


if __name__ == "__main__":
    main()
