#!/usr/bin/env python3
"""Create separate runtime and validation plots for the berni probe snapshot.

Run from the repository root: python analysis/plot_final_probe.py
The berni research data are read from the local origin/berni Git ref.
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final_probe_plots"
BRANCH_DIR = "solution/research/probe"
MODEL_LABEL = "simulation model"
COLORS = ["#4C78A8", "#F58518", "#54A24B"]


def branch_text(name: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"origin/berni:{BRANCH_DIR}/{name}"],
        cwd=ROOT,
        text=True,
    )


def branch_csv(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(branch_text(name))))


def read_jsonl(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Some research logs append plain-text progress summaries.
    return rows


def save_boxplot(data: list[list[float]], labels: list[str], title: str,
                 ylabel: str, path: Path, *, limit: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    boxes = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                       medianprops={"color": "#222222", "linewidth": 2})
    for box, color in zip(boxes["boxes"], COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.78)
    if limit is not None:
        ax.axhline(limit, color="#D62728", linestyle="--", linewidth=1.8,
                   label=f"{limit:g} second limit")
        ax.legend(frameon=False, loc="upper right")
        ax.set_ylim(bottom=0, top=limit * 1.08)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with (ROOT / "runtime-data.csv").open(newline="") as f:
        runtime_rows = list(csv.DictReader(f))
    threshold_data = []
    for threshold in (16, 64, 512):
        threshold_data.append([
            float(r["duration_s"]) for r in runtime_rows
            if int(r["threshold"]) == threshold and r["status"] == "success"
        ])
    # Challenge run times span many orders of magnitude, so use a log axis.
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    bp = ax.boxplot(threshold_data, labels=["16", "64", "512"], patch_artist=True,
                    showfliers=False, medianprops={"color": "#222222", "linewidth": 2})
    for box, color in zip(bp["boxes"], COLORS):
        box.set_facecolor(color); box.set_alpha(0.78)
    ax.set_yscale("log")
    ax.set_title(f"{MODEL_LABEL}: measured runtime by threshold", fontsize=14, weight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Measured runtime (seconds, log scale)")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.savefig(OUT / "threshold_runtimes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    probes = read_jsonl(branch_text("probe_variants.jsonl"))
    # The final NN sweep selects c4b0.5; use its per-circuit inference feature timing.
    analysis = [float(r["c4b0.5_probe_seconds"]) for r in probes
                if "c4b0.5_probe_seconds" in r and not r.get("error")]
    extraction = [float(r["parse_seconds"]) for r in probes
                  if "parse_seconds" in r and not r.get("error")]
    n = min(len(extraction), len(analysis))
    extraction, analysis = extraction[:n], analysis[:n]
    total = [a + b for a, b in zip(extraction, analysis)]
    save_boxplot([extraction, analysis, total], ["Extraction", "Analysis", "Total"],
                 f"{MODEL_LABEL}: per circuit processing time",
                 "Time (seconds)", OUT / "model_runtime_components.png", limit=15)

    # Raw final NN OOF predictions are not present in the branch snapshot. Use the
    # bundled row-level reconstructed Bernie evaluation to make residual/error diagnostics,
    # while labeling the provenance plainly in each figure.
    final_sweep = read_jsonl(branch_text("round2_results.jsonl"))
    candidate = next(r for r in final_sweep
                     if r.get("probe") == "c4b0.5" and r.get("width") == 256
                     and r.get("depth") == 2 and r.get("wd") == 0.01)
    score = float(candidate["cv5"])
    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    ax.bar([MODEL_LABEL], [score], color=COLORS[0], width=0.55)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Grouped 5-fold challenge score (%)")
    ax.set_title(f"{MODEL_LABEL}: validation score (compact + probe MLP)", fontsize=14, weight="bold")
    ax.text(0, score + 1.1, f"{score:.2f}%", ha="center", weight="bold")
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(OUT / "validation_score.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    eval_rows = list(csv.DictReader((ROOT / "reports/shared-evaluation/structure_grouped/berni-paired-errors.csv").open(newline="")))
    actual = np.array([float(r["reference_seconds"]) for r in eval_rows])
    predicted = np.array([float(r["berni_pred_seconds"]) for r in eval_rows])
    residual = predicted - actual
    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)
    ax.scatter(predicted, residual, s=13, alpha=0.42, color=COLORS[0], edgecolors="none")
    ax.axhline(0, color="#D62728", linestyle="--", linewidth=1.5)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Predicted runtime (seconds, log scale)")
    ax.set_ylabel("Residual: predicted − measured (seconds, symlog)")
    ax.set_title(f"{MODEL_LABEL}: R² residual diagnostic\n(row-level reconstructed probe evaluation)", fontsize=13, weight="bold")
    ax.grid(alpha=0.22, which="both")
    fig.savefig(OUT / "r2_residual_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ratio = np.maximum(predicted / actual, actual / predicted)
    counts = [int(np.sum(ratio < 2)),
              int(np.sum((ratio >= 2) & (ratio < 10))),
              int(np.sum(ratio >= 10))]
    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    labels = ["<2× error", "2–10× error", ">10× error"]
    bars = ax.bar(labels, counts, color=COLORS, width=0.6)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Validation batches (circuit-threshold rows)")
    ax.set_title(f"{MODEL_LABEL}: validation error batches\n(row-level reconstructed probe evaluation)", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(OUT / "validation_error_histogram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved five separate figures to {OUT}")
    print(f"Probe rows: {n}; runtime component maximum total: {max(total):.4f} s")
    print(f"Validation diagnostic source rows: {len(eval_rows)}; error bins: <2x={counts[0]}, 2–10x={counts[1]}, >10x={counts[2]}")
    print("Residual/error plots use the available reconstructed probe evaluation rows; final NN OOF predictions are absent from the probe snapshot.")


if __name__ == "__main__":
    main()
