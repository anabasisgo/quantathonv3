# Quantathon — Submission Harness

Predict how long each circuit takes to simulate, straight from the `.qasm` file.
This harness produces the **one file you send back to us**.

## Setup (once)

```bash
pip install -r requirements.txt
```

## What you edit

**Only `model.py`.** Implement these methods:

| Method | Job | Called |
|---|---|---|
| `RuntimeModel()` | load your trained model | once |
| `featurize(qasm_text)` | parse a circuit into features | once per circuit |
| `predict(features, threshold)` | return **predicted seconds** (one number) | once per (circuit, threshold) |

There is no timeout flag — if you think a run will time out, just predict a duration
**≥ the 4-hour cap (14400 s)**.

`model.py` and `artifacts/runtime-model.json` contain the fitted runtime model.
It combines 123 structural/threshold inputs with 22 mock-simulation inputs.
The bounded Clifford surrogate tracks approximate entanglement through a
circuit prefix; `predict` derives bond-cost features for the requested threshold.
Inference requires only the Python standard library. QASM decompression may use
the optional `zstandard` package as before.

The simulator processes at most 20,000 expanded gates, 8 million source
characters and 512 qubits. Truncated inputs carry coverage/incomplete features.
Its 10-second cooperative deadline also respects the time left after structural
parsing, reserving one second of the 15-second budget. Measured integration
timings and limitations are in
[the implementation report](../reports/mock-simulation/implementation/README.md).

To refit the evaluated recipe on the existing training library:

```bash
uv run --offline --cache-dir /tmp/quantathon-uv-cache \
  --with scikit-learn==1.9.1 --with xgboost-cpu==3.4.1 \
  python quantathon-harness/train_mock.py
```

Run this command from the repository root. It checks live features against the
recorded experiment and portable predictions against scikit-learn before
replacing the artifact. The previous source and artifact are saved in
`reports/mock-simulation/pre-integration/`. Older artifacts remain loadable.

**Caps:** `featurize` and `predict` must each run in **≤ 15 s per circuit**. The
harness times you and warns on anything over.

## What's included

| Path | Contents |
|---|---|
| `circuits/` | the training circuits, one `<id>.qasm.zst` per circuit |
| `runtime-data.csv` | training labels, one row per `(circuit, threshold)` run |

The circuits are `zstd`-compressed. `run.py` decompresses them for you, so you
don't need raw files to run the harness. To get raw `.qasm` files for exploring
and training, see
[Getting the raw `.qasm` files](../README.md#getting-the-raw-qasm-files) in the
main README. Decompress them **outside** `circuits/`, or the harness will read
each circuit twice.

## Run it

1. `circuits/` already holds the training circuits. In the final hours we DM you
   the hold-out circuits: put them in `circuits/` too (`.qasm` or `.qasm.zst`,
   subfolders are fine). Only hold-out runs are scored, so you can remove the
   training circuits first to make the run faster.
2. Generate your submission:

```bash
python run.py --team "Your Team Name"
```

This writes **`submission.csv`**:

```
team,filename,threshold,pred_duration_s,parse_s,predict_s
```

3. **DM `submission.csv` back to us.** That's your entry.

## Check your score first (optional but recommended)

You have the training labels (`runtime-data.csv`). Run `run.py` on the training
circuits, then score yourself with the *exact* metric we use:

```bash
python score.py --pred submission.csv --labels runtime-data.csv
```

## How the automated third is scored

Your total score is **1/3 automated + 2/3 subjective** (presentation, novelty,
process, etc. — judged live, five equally-weighted categories). The automated third
is **pure duration accuracy**, per `(circuit, threshold)`:

```
score = max(0, 1 − |log10(pred / actual)| / 2)      # exact = 1.0, off by 10× = 0.5
```

Runtimes span seconds to hours, so accuracy is measured in **log scale** — being
2× off costs the same whether the run is 1 second or 1 hour. Timeout circuits are
scored against the 4-hour cap, so predicting ≥ cap earns full credit on them and
predicting seconds for one is penalized like any other large miss.

Your score is the average over **every** hold-out run. A run missing from your
`submission.csv` scores 0, so make sure every circuit gets a prediction.

## Rules

- Build your own parser; no starter feature list is provided.
- You must be able to explain every part of your submission.
- Don't try to recover the source algorithm from filenames/metadata.
