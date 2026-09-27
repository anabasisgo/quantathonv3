# NN A: training and evaluation

The submission code lives in `quantathon-harness/` (`model.py` + the `nnrt/` package + `artifacts/`).
This folder rebuilds the artifacts and measures them. Run everything from the repository root.

| Step | Command | Output |
|---|---|---|
| 1. Features | `python training/build_features.py` | `training/features.csv` (the submission's own `featurize()` on all 532 circuits) |
| 2. Validate | `python training/cross_validate.py` | grouped 5-fold score and error stats; `training/cv_predictions.csv` |
| 3. Train | `python training/train.py` | `quantathon-harness/artifacts/nn_a.npz`, `nn_a.json`; checks numpy/torch parity |
| 4. Time | `python training/time_pipeline.py` | per-stage timing over all circuits; `training/timing.csv` |
| 5. Harness | `cd quantathon-harness && python run.py --team T --circuits ../training_circuits` then `python score.py --pred submission.csv --labels ../runtime-data.csv` | in-sample check of the full path |

Install training dependencies with `pip install -r training/requirements.txt`; the harness itself needs only
`numpy` and `zstandard` (`quantathon-harness/requirements.txt`). Training uses a CUDA GPU when available.

## Pipeline

`featurize(qasm)` → 44 features, each stage bounded:

1. **Compact structure (26)**: the team's v3 extractor (`nnrt/extractor*.py`, `graph.py`, vendored) → counts, depth,
   interaction graph, cut/span statistics in native and RCM order, SupermarQ-style metrics (`nnrt/compact.py`).
   Budget 6 s; values the extractor cannot certify are left missing.
2. **Probe parse** (`nnrt/qasm_ops.py`): the first 2 M characters / 150 k operations → 1- and 2-qubit gate list.
3. **Structural bond bound (10)** (`nnrt/probe.py`): upper bound on each cut's bond dimension → cost estimates per threshold.
4. **Truncated MPS probe (8)** (`nnrt/probe.py`): bond cap 16, deterministic work budget (≈1 s on the reference
   machine; hard 3 s wall-clock stop) → saturation, bond size, peak entropy, truncated weight, work, coverage.

`predict(features, threshold)`: signed-log1p + median-impute + standardize, 4 threshold inputs, then the mean of 5
MLPs (1024-512-256, GELU) on log10 seconds (`nnrt/network.py`, numpy only). Training target: log10 runtime, with the
33 timeouts and one over-cap success at the 14,400 s cap. `folds.csv` holds the team's structural 5-fold split
(all thresholds of a circuit and its structural duplicates are held out together).

## Recorded baseline (26 Sep 2026, 2-core cloud VM, CPU only)

These measurements predate the parser and probe optimizations; rerun the timing
script on the target machine for current per-stage numbers.

- **Grouped 5-fold CV** (`cross_validate.py`): score **94.81** (t=16 95.44, t=64 94.70, t=512 94.18); R² on log10 runtime 0.960;
  91.5% of completed runs within 2×, 99.0% within 10×; 10 of 33 timeouts predicted at the cap, 1 false timeout.
- **Official harness on all 532 training circuits** (`run.py`, single process): no crashes, no cap warnings.
  Per circuit: featurize median 1.12 s, p99 4.57 s, max 8.41 s; each predict call ≤ 0.055 s; featurize + 3 predicts max 8.46 s.
  `score.py` in-sample: 98.81% (trained on these circuits, so this checks the plumbing, not generalization).
- **Stage maxima** (from `build_features.py`, 2 parallel workers): compact extractor 6.16 s, probe parse 1.63 s,
  bond bound 0.47 s, probe simulation 1.59 s. Numpy vs torch prediction gap: < 1e-6 log10 units.
