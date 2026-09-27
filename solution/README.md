# Quantum Rings runtime predictor

The ready participant solution is [`../quantathon-harness/model.py`](../quantathon-harness/model.py). It embeds the fitted v5 model and its QASM feature parser in one standard-library Python file. The organizer's `run.py`, `score.py`, requirements, training circuits and labels remain unchanged. This branch does not contain an organizer submission or hidden-set result.

## Generate and check predictions

Install the harness dependency from `quantathon-harness/requirements.txt`. Use a directory containing **only** the organizer's released holdout circuits. The harness accepts `.qasm` and `.qasm.zst` recursively; duplicate basenames, including compressed/plain copies, create ambiguous output keys.

From the repository root, run:

```bash
python solution/preflight.py --circuits ABSOLUTE_PATH_TO_HOLDOUT --report holdout_input_check.json
cd quantathon-harness
python run.py --team "YOUR ACTUAL TEAM NAME" --circuits ABSOLUTE_PATH_TO_HOLDOUT --out submission.csv
cd ..
python solution/preflight.py --circuits ABSOLUTE_PATH_TO_HOLDOUT --csv quantathon-harness/submission.csv --report holdout_submission_check.json
```

Check that both preflight reports say `passed` and that the harness reported no read or decompression failures. The output must have one row per circuit at each threshold (16, 64, 512), with positive finite predictions and parser/inference times at most 15 seconds. If organizer labels become available, score the CSV with `quantathon-harness/score.py`. Do not score an unlabeled hidden set using training labels.

## What was measured

- The selected v5 model scored **89.88%** on an internal 107-circuit holdout (301 labeled runs) through the supplied harness and scorer. Earlier exploratory analysis had seen these circuits; this is not an untouched organizer test.
- Its all-data refit uses 532 training circuits and 1,497 labeled runs. A practice run emitted 1,596 predictions (three per circuit), with maximum parsing time 7.4839 seconds and prediction time 0.0032 seconds on the local machine. Its 96.49% score on known training labels is in-sample.
- Later five-fold grouped research found a forest/SVR blend at **91.19%** versus **90.30%** for this v5 reference on those folds. That blend lost ground on several source-exclusion stress tests, so it has **not** replaced the ready v5 model. None of these figures is an organizer hidden-set score.

The model is self-contained for inference. `BUILD_MANIFEST.json` identifies the fitted bundle and hashes. Research training artifacts and scripts are retained locally; the reports in `research/` document their methods and limits but this compact branch does not claim to reproduce model fitting from scratch. The existing `training_circuits/` and `runtime-data.csv` remain in their original locations rather than being copied into `solution/`.

## Research notes

- [V5 model selection, failure cases and holdout](research/v5/MODEL_REPORT.md)
- [V5 data and deployment checks](research/v5/DATA_DEPLOYMENT_REPORT.md)
- [V6 decision](research/v6/DECISION.md), [model comparison](research/v6/MODELING_REPORT.md), and [feature analysis](research/v6/FEATURE_REPORT.md)

The published files are a curated snapshot: reports and their linked figures, the runnable v5 model, the input/output preflight tool, and build hashes. Checkpoints, alternate fitted models, all-fold predictions, intermediate feature tables, dependency caches, and redundant HTML copies are omitted. The reports may mention those local artifacts by their original names; they are evidence references, not files in this branch.
