# Quantathon runtime predictor

This repository contains the NN A model and the organizer's submission harness.
The harness loads a trained ensemble, extracts features from each QASM circuit,
and writes one predicted duration for every requested threshold.

## Run the model

Install the inference dependencies from `quantathon-harness/requirements.txt`.
From the repository root, run:

```bash
python quantathon-harness/run.py --team "Your Team" --circuits training_circuits --out quantathon-harness/submission.csv
python quantathon-harness/score.py --pred quantathon-harness/submission.csv --labels runtime-data.csv
```

The score command uses the provided training labels; it is an in-sample check,
not a held-out evaluation. `run.py` also accepts a different circuit directory
for the final submission. Do not place decompressed copies beside the `.qasm.zst`
files, or the harness will read each circuit twice.

## What each part is for

| Path | Purpose |
|---|---|
| `quantathon-harness/run.py`, `score.py` | Organizer's runner and local scoring script |
| `quantathon-harness/model.py`, `nnrt/` | Feature extraction and model inference |
| `quantathon-harness/artifacts/nn_a.npz`, `nn_a.json` | Required trained weights and feature metadata |
| `training/` | Rebuild features, cross-validate, train, and measure the model; see [training/README.md](training/README.md) |
| `training_circuits/`, `runtime-data.csv` | Training circuits and runtime labels |
| `training/folds.csv` | Group assignments needed to reproduce validation |
| `training/features.csv` | Feature table used to retrain without extracting all circuits again |

Only the harness code, `model.py`, `nnrt/`, and the two files in `artifacts/`
are needed for inference. The training data and scripts are retained so the
model and reported validation can be reproduced.
